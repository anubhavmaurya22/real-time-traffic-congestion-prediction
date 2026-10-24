"""
Custom routing for arbitrary origin/destination in Bangalore.

Fetches up to 3 alternative driving routes from OSRM, samples points
every ~200 m along each route, computes ETA using the spatial speed
estimator (model-predicted speeds interpolated to arbitrary points),
and recommends the route with the lowest predicted ETA.

Used by POST /route/custom in api/api_bangalore.py.
"""

import math
import httpx


# ── Geometry helpers ──────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def sample_points_along_line(coords, interval_m=200):
    """
    Given a list of [lng, lat] GeoJSON coordinates, return a list of
    (lat, lng) tuples sampled approximately every interval_m metres.
    Always includes the first and last points.
    """
    if not coords or len(coords) < 2:
        return [(coords[0][1], coords[0][0])] if coords else []

    sampled = [(coords[0][1], coords[0][0])]  # first point (lat, lng)
    accumulated = 0.0

    for i in range(1, len(coords)):
        lat1, lng1 = coords[i - 1][1], coords[i - 1][0]
        lat2, lng2 = coords[i][1], coords[i][0]
        seg_dist = haversine_m(lat1, lng1, lat2, lng2)
        accumulated += seg_dist

        if accumulated >= interval_m:
            sampled.append((lat2, lng2))
            accumulated = 0.0

    # Always include last point
    last = (coords[-1][1], coords[-1][0])
    if sampled[-1] != last:
        sampled.append(last)

    return sampled


# ── OSRM client ──────────────────────────────────────────────────────────────

OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"


async def fetch_osrm_routes(origin_lat, origin_lng, dest_lat, dest_lng,
                            timeout_s=8.0, max_alternatives=3):
    """
    Fetch up to max_alternatives driving routes from OSRM.
    Returns list of route dicts with 'geometry' (GeoJSON), 'distance' (m),
    'duration' (s).
    Raises httpx.HTTPError or ValueError on failure.
    """
    coords_str = f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
    url = f"{OSRM_BASE}/{coords_str}"
    params = {
        "alternatives": "true",
        "geometries": "geojson",
        "overview": "full",
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != "Ok":
        raise ValueError(f"OSRM error: {data.get('code')} — {data.get('message', '')}")

    routes = data.get("routes", [])
    if not routes:
        raise ValueError("OSRM returned no routes")

    return routes[:max_alternatives]


# ── Route scoring ─────────────────────────────────────────────────────────────

def score_route(route_geojson, speed_estimator, predicted_speeds, day_offset,
                training_mean_speed, sample_interval_m=200):
    """
    Score a single OSRM route.

    Distance scaling: the sum of chord distances between sampled points
    under-estimates real road distance. We scale by (OSRM distance / chord sum)
    so ETA and avg_speed are consistent with the OSRM road distance.

    Returns:
        dict with eta_pred_min, eta_avg_speed_min, avg_speed_kmh, distance_km,
        geometry (GeoJSON LineString), coverage_warning.
    """
    coords = route_geojson["geometry"]["coordinates"]
    distance_m = route_geojson["distance"]
    distance_km = distance_m / 1000.0

    sampled = sample_points_along_line(coords, interval_m=sample_interval_m)

    # Sum of chord (great-circle) distances between sampled points
    chord_sum_km = 0.0
    # Accumulate time weighted by chord segments (before scaling)
    total_time_pred_h = 0.0
    total_time_avg_h = 0.0
    any_coverage_warning = False

    for i in range(len(sampled) - 1):
        lat1, lng1 = sampled[i]
        lat2, lng2 = sampled[i + 1]
        seg_km = haversine_km(lat1, lng1, lat2, lng2)

        if seg_km < 0.001:
            continue  # skip near-zero segments

        chord_sum_km += seg_km

        # Speed at midpoint of segment
        mid_lat = (lat1 + lat2) / 2
        mid_lng = (lng1 + lng2) / 2
        speed_pred, cov_warn = speed_estimator.speed_at(
            mid_lat, mid_lng, predicted_speeds, day_offset
        )
        if cov_warn:
            any_coverage_warning = True

        total_time_pred_h += seg_km / speed_pred
        total_time_avg_h += seg_km / training_mean_speed

    # Scale factor: OSRM road distance / chord distance sum
    # This corrects for the fact that straight-line segments undercount
    # actual road distance. Both ETA values are scaled equally.
    scale = distance_km / chord_sum_km if chord_sum_km > 0 else 1.0

    eta_pred_min = total_time_pred_h * scale * 60.0
    eta_avg_speed_min = total_time_avg_h * scale * 60.0
    avg_speed = distance_km / (total_time_pred_h * scale) if total_time_pred_h > 0 else training_mean_speed

    return {
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "distance_km": round(distance_km, 2),
        "eta_pred_min": round(eta_pred_min, 1),
        "eta_avg_speed_min": round(eta_avg_speed_min, 1),
        "avg_speed_kmh": round(avg_speed, 1),
        "coverage_warning": any_coverage_warning,
    }


def score_all_routes(osrm_routes, speed_estimator, predicted_speeds,
                     day_offset, training_mean_speed):
    """
    Score all OSRM routes and pick the recommended one.

    time_saved_min: ETA of OSRM's default first route minus ETA of the
    recommended route, both scored with predicted speeds. If the recommended
    route IS the default route, time_saved = 0.

    Returns the full response dict.
    """
    scored = []
    for i, route in enumerate(osrm_routes):
        result = score_route(
            route, speed_estimator, predicted_speeds, day_offset,
            training_mean_speed
        )
        result["id"] = f"route_{i}"
        scored.append(result)

    # Recommend route with lowest predicted ETA
    if scored:
        best_idx = min(range(len(scored)), key=lambda i: scored[i]["eta_pred_min"])
        for i, r in enumerate(scored):
            r["recommended"] = (i == best_idx)

        # Default route is OSRM's first route (index 0)
        default_eta = scored[0]["eta_pred_min"]
        best_eta = scored[best_idx]["eta_pred_min"]

        # Time saved = default route ETA - recommended route ETA
        # If the recommended route IS the default, time_saved = 0
        if best_idx == 0:
            time_saved = 0.0
            improvement_pct = 0.0
        else:
            time_saved = default_eta - best_eta
            improvement_pct = (time_saved / default_eta * 100
                               if default_eta > 0 else 0.0)

        default_route_eta = default_eta
    else:
        time_saved = 0.0
        improvement_pct = 0.0
        default_route_eta = 0.0

    any_coverage = any(r["coverage_warning"] for r in scored)

    return {
        "routes": scored,
        "default_route_eta_min": round(default_route_eta, 1),
        "time_saved_min": round(time_saved, 1),
        "improvement_pct": round(improvement_pct, 1),
        "coverage_warning": any_coverage,
    }
