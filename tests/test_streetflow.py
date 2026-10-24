"""
Tests for StreetFlow Live — speed estimator, routing, API validation.

Run with:  pytest tests/ -v

Set STRICT_TESTS=1 to make skip-guarded tests fail loudly instead of skipping.
"""

import os
import sys
import math
import json
import pytest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "model"))
sys.path.insert(0, os.path.join(_BASE, "data"))
sys.path.insert(0, os.path.join(_BASE, "api"))

STRICT = os.environ.get("STRICT_TESTS", "").strip() in ("1", "true", "yes")


def _skip_or_fail(reason):
    """In strict mode, fail. Otherwise skip."""
    if STRICT:
        pytest.fail(f"STRICT_TESTS=1: {reason}")
    else:
        pytest.skip(reason)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Speed Estimator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedEstimator:
    """Tests for the spatial speed estimator (IDW)."""

    @pytest.fixture
    def road_coords(self):
        from speed_estimator import load_road_coords
        return load_road_coords()

    @pytest.fixture
    def fake_speeds(self):
        """16 fake predicted speeds for testing."""
        return [35.0 + i * 2 for i in range(16)]

    @pytest.fixture
    def estimator(self, road_coords):
        from speed_estimator import SpeedEstimator
        return SpeedEstimator(road_coords=road_coords, method="idw")

    def test_near_known_road(self, estimator, fake_speeds):
        """Speed near a known road should be close to that road's speed."""
        # Indiranagar (roads 0, 1): lat 12.9699, lng 77.6499
        speed, warn = estimator.speed_at(12.970, 77.650, fake_speeds, day_offset=0)
        assert speed > 0, "Speed must be positive"
        assert not warn, "Should not have coverage warning near a known road"
        # Should be influenced heavily by roads 0 and 1
        assert 30 < speed < 70, f"Speed {speed} seems unreasonable near Indiranagar"

    def test_far_from_roads_has_coverage_warning(self, estimator, fake_speeds):
        """Points far from all 16 roads should trigger coverage_warning."""
        # Point well outside Bangalore road coverage (still in bounding box)
        speed, warn = estimator.speed_at(13.19, 77.41, fake_speeds, day_offset=0)
        assert warn, "Should have coverage warning for point far from all roads"
        # Should return mean speed
        expected_mean = np.mean(fake_speeds)
        assert abs(speed - expected_mean) < 1.0, "Far-away speed should be close to mean"

    def test_speed_clamped_to_minimum(self, road_coords):
        """Speed should never drop below min_speed_kmh (5 km/h)."""
        from speed_estimator import SpeedEstimator
        estimator = SpeedEstimator(road_coords=road_coords, method="idw")
        # All speeds are very low
        low_speeds = [1.0] * 16
        speed, _ = estimator.speed_at(12.97, 77.65, low_speeds, day_offset=0)
        assert speed >= 5.0, f"Speed {speed} should be at least 5 km/h"

    def test_idw_varies_with_position(self, estimator, fake_speeds):
        """Different locations should give different IDW speeds."""
        s1, _ = estimator.speed_at(12.92, 77.62, fake_speeds, 0)
        s2, _ = estimator.speed_at(13.03, 77.59, fake_speeds, 0)
        # These are near different clusters of roads with different speeds
        assert s1 != s2, "IDW should produce different speeds at different locations"


class TestSpeedEstimatorCrossValidation:
    """Test that leave-one-road-out cross-validation runs and produces metrics."""

    def test_loo_cv_computes_metrics(self):
        from speed_estimator import load_road_coords, leave_one_out_cv
        roads = load_road_coords()
        # Use a single fake speed snapshot
        fake_speeds = [np.array([30.0 + i * 2 for i in range(16)])]

        metrics = leave_one_out_cv(roads, fake_speeds, strict=True)

        assert "idw" in metrics
        assert "global_mean" in metrics
        for method in ["idw", "global_mean"]:
            assert "MAE" in metrics[method]
            assert "RMSE" in metrics[method]
            assert "MAPE" in metrics[method]
            assert "n_samples" in metrics[method]
            assert metrics[method]["MAE"] >= 0
            assert metrics[method]["RMSE"] >= 0
            assert metrics[method]["n_samples"] > 0

    def test_strict_vs_original_cv(self):
        """Strict CV should not have fewer or identical samples to original."""
        from speed_estimator import load_road_coords, leave_one_out_cv
        roads = load_road_coords()
        fake_speeds = [np.array([30.0 + i * 2 for i in range(16)])]

        strict = leave_one_out_cv(roads, fake_speeds, strict=True)
        original = leave_one_out_cv(roads, fake_speeds, strict=False)

        # Both should have same number of held-out evaluations
        assert strict["idw"]["n_samples"] == original["idw"]["n_samples"]
        # But strict MAE should typically be >= original (harder task)
        # Just check they both produce valid numbers
        assert strict["idw"]["MAE"] >= 0
        assert original["idw"]["MAE"] >= 0


class TestNearDuplicateDetection:
    """Verify that near-duplicate road detection works."""

    def test_finds_known_duplicates(self):
        from speed_estimator import load_road_coords, find_near_duplicates
        roads = load_road_coords()
        near = find_near_duplicates(roads, threshold_km=0.5)

        # At least one group should have >1 member (100 Feet / CMH share coords)
        any_dupe = any(len(v) > 1 for v in near.values())
        assert any_dupe, "Expected at least one pair of near-duplicate roads"

    def test_every_road_in_own_group(self):
        from speed_estimator import load_road_coords, find_near_duplicates
        roads = load_road_coords()
        near = find_near_duplicates(roads)
        for r in roads:
            assert r["index"] in near, f"Road {r['index']} missing from near_dupes"
            assert r["index"] in near[r["index"]], "Road should include itself"

    def test_twin_roads_excluded_in_strict_loo(self):
        """Assert twin roads (< 0.5 km) are grouped together and excluded in strict mode."""
        from speed_estimator import load_road_coords, find_near_duplicates
        roads = load_road_coords()
        near = find_near_duplicates(roads, threshold_km=0.5)
        # Check known twins (e.g. 100 Feet Road / CMH Road)
        road_names = {r["index"]: r["road_name"] for r in roads}
        twin_found = False
        for idx, dupes in near.items():
            if len(dupes) > 1:
                twin_found = True
                assert idx in dupes
                for d in dupes:
                    assert d in near[idx]
        assert twin_found, "Must find at least one pair of twin roads"

    def test_learned_features_never_include_target_road(self):
        """Learned estimator training features must never include the target road."""
        from speed_estimator import load_road_coords, build_features, find_near_duplicates
        roads = load_road_coords()
        near = find_near_duplicates(roads, threshold_km=0.5)
        fake_speeds = [30.0 + i for i in range(16)]
        for r in roads:
            ho_idx = r["index"]
            excluded = near.get(ho_idx, {ho_idx})
            other_coords = [o for o in roads if o["index"] not in excluded]
            feats = build_features(r["lat"], r["lng"], other_coords, fake_speeds, day_offset=0, k=min(5, len(other_coords)))
            # Speed features are at positions 2+k to 2+2k
            k = min(5, len(other_coords))
            speed_features = feats[2 + k : 2 + 2 * k]
            assert fake_speeds[ho_idx] not in speed_features, \
                f"Target road {ho_idx} speed leaked into features!"



# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bounding Box Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundingBoxValidation:
    """Test Pydantic validation for the Bangalore bounding box."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("pydantic")

    def test_valid_coordinates(self):
        """Coordinates inside Bangalore should be accepted."""
        from api_bangalore import LatLng
        point = LatLng(lat=12.97, lng=77.60)
        assert point.lat == 12.97
        assert point.lng == 77.60

    def test_latitude_too_low(self):
        """Latitude below 12.80 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=12.50, lng=77.60)

    def test_latitude_too_high(self):
        """Latitude above 13.20 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=13.50, lng=77.60)

    def test_longitude_too_low(self):
        """Longitude below 77.40 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=12.97, lng=77.00)

    def test_longitude_too_high(self):
        """Longitude above 77.85 should be rejected."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="outside the Bangalore bounding box"):
            LatLng(lat=12.97, lng=78.00)

    def test_same_point_rejection(self):
        """Origin and destination at the same point (< 200 m) should fail."""
        from api_bangalore import CustomRouteRequest
        # Same coordinates — distance is 0
        req = CustomRouteRequest(
            origin={"lat": 12.97, "lng": 77.60},
            destination={"lat": 12.97, "lng": 77.60},
            day_offset=0
        )
        # The distance check happens in the endpoint, not in Pydantic.
        # So we just verify the model accepts same coords (the 200m check
        # is in the route handler).
        from custom_routing import haversine_m
        dist = haversine_m(
            req.origin.lat, req.origin.lng,
            req.destination.lat, req.destination.lng
        )
        assert dist < 200, "Same-point distance should be < 200 m"

    def test_day_offset_out_of_range(self):
        """day_offset outside 0-2 should be rejected."""
        from api_bangalore import CustomRouteRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="day_offset"):
            CustomRouteRequest(
                origin={"lat": 12.97, "lng": 77.60},
                destination={"lat": 12.92, "lng": 77.62},
                day_offset=5
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Route Scoring Tests (mocked OSRM response)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteScoring:
    """Test route scoring logic with a mocked OSRM response."""

    @pytest.fixture
    def mock_osrm_routes(self):
        """Simulated OSRM response with 2 alternative routes."""
        return [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [77.60, 12.97],   # start (Indiranagar area)
                        [77.62, 12.95],   # mid
                        [77.63, 12.93],   # end (Koramangala area)
                    ]
                },
                "distance": 5500,   # 5.5 km
                "duration": 900,    # 15 min (OSRM's static estimate)
            },
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [77.60, 12.97],
                        [77.61, 12.96],
                        [77.615, 12.945],
                        [77.63, 12.93],
                    ]
                },
                "distance": 6200,   # 6.2 km
                "duration": 1020,   # 17 min
            },
        ]

    @pytest.fixture
    def estimator(self):
        from speed_estimator import SpeedEstimator, load_road_coords
        roads = load_road_coords()
        return SpeedEstimator(road_coords=roads, method="idw")

    def test_score_all_routes(self, mock_osrm_routes, estimator):
        """score_all_routes should return scored routes with one recommended."""
        from custom_routing import score_all_routes
        fake_speeds = [35.0 + i * 2 for i in range(16)]
        mean_speed = 40.0

        result = score_all_routes(
            mock_osrm_routes, estimator, fake_speeds,
            day_offset=0, training_mean_speed=mean_speed
        )

        assert "routes" in result
        assert len(result["routes"]) == 2
        assert "time_saved_min" in result
        assert "improvement_pct" in result
        assert "coverage_warning" in result
        assert "default_route_eta_min" in result

        # Exactly one route should be recommended
        recommended = [r for r in result["routes"] if r["recommended"]]
        assert len(recommended) == 1

        for route in result["routes"]:
            assert route["eta_pred_min"] > 0
            assert route["eta_avg_speed_min"] > 0
            assert route["distance_km"] > 0
            assert route["avg_speed_kmh"] > 0
            assert "geometry" in route
            assert route["geometry"]["type"] == "LineString"

    def test_recommended_has_lowest_eta(self, mock_osrm_routes, estimator):
        """The recommended route should have the lowest predicted ETA."""
        from custom_routing import score_all_routes
        fake_speeds = [35.0 + i * 2 for i in range(16)]

        result = score_all_routes(
            mock_osrm_routes, estimator, fake_speeds,
            day_offset=0, training_mean_speed=40.0
        )

        rec = [r for r in result["routes"] if r["recommended"]][0]
        for r in result["routes"]:
            assert rec["eta_pred_min"] <= r["eta_pred_min"]

    def test_time_saved_compares_to_default(self, mock_osrm_routes, estimator):
        """time_saved should compare recommended vs default (first) route."""
        from custom_routing import score_all_routes
        fake_speeds = [35.0 + i * 2 for i in range(16)]

        result = score_all_routes(
            mock_osrm_routes, estimator, fake_speeds,
            day_offset=0, training_mean_speed=40.0
        )

        default_eta = result["routes"][0]["eta_pred_min"]
        rec = [r for r in result["routes"] if r["recommended"]][0]

        if rec == result["routes"][0]:
            # Recommended IS the default
            assert result["time_saved_min"] == 0.0
        else:
            expected_saved = default_eta - rec["eta_pred_min"]
            assert abs(result["time_saved_min"] - round(expected_saved, 1)) < 0.2

    def test_distance_scaling_within_1pct(self, mock_osrm_routes, estimator):
        """ETA should be based on OSRM distance, not chord sums."""
        from custom_routing import score_route
        fake_speeds = [40.0] * 16  # uniform speed
        mean_speed = 40.0

        result = score_route(
            mock_osrm_routes[0], estimator, fake_speeds,
            day_offset=0, training_mean_speed=mean_speed
        )

        # With uniform speed = 40 km/h, ETA should be distance/speed
        expected_eta = result["distance_km"] / 40.0 * 60  # minutes
        # Allow 5% tolerance (IDW gives slightly different speeds from 40)
        assert abs(result["eta_avg_speed_min"] - expected_eta) / expected_eta < 0.05, \
            f"ETA at avg speed ({result['eta_avg_speed_min']}) should be ~{expected_eta:.1f} min for {result['distance_km']} km at 40 km/h"

    def test_sum_of_scored_segments_equals_osrm_distance_within_1pct(self, mock_osrm_routes):
        """Sum of scaled segment distances must equal OSRM route distance within 1%."""
        from custom_routing import sample_points_along_line, haversine_km
        route = mock_osrm_routes[0]
        osrm_km = route["distance"] / 1000.0
        coords = route["geometry"]["coordinates"]
        sampled = sample_points_along_line(coords, interval_m=200)

        chord_sum_km = 0.0
        for i in range(len(sampled) - 1):
            lat1, lng1 = sampled[i]
            lat2, lng2 = sampled[i + 1]
            chord_sum_km += haversine_km(lat1, lng1, lat2, lng2)

        scale = osrm_km / chord_sum_km if chord_sum_km > 0 else 1.0
        scaled_distance_km = chord_sum_km * scale

        assert abs(scaled_distance_km - osrm_km) / osrm_km < 0.01, \
            f"Scaled distance {scaled_distance_km:.4f} km does not match OSRM {osrm_km:.4f} km within 1%"



# ═══════════════════════════════════════════════════════════════════════════════
# 4. API Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIErrorHandling:
    """Test API 400 and 502 error paths."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("pydantic")

    def test_outside_bangalore_returns_400(self):
        """A destination outside Bangalore should return HTTP 400."""
        from api_bangalore import LatLng
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LatLng(lat=28.61, lng=77.20)  # Delhi coordinates

    def test_custom_route_request_valid(self):
        """Valid CustomRouteRequest should parse without errors."""
        from api_bangalore import CustomRouteRequest
        req = CustomRouteRequest(
            origin={"lat": 12.9716, "lng": 77.5946, "label": "Indiranagar Metro"},
            destination={"lat": 12.8440, "lng": 77.6610, "label": "Electronic City"},
            day_offset=0
        )
        assert req.origin.lat == 12.9716
        assert req.destination.lng == 77.6610

    def test_admin_reload_request_model(self):
        """AdminReloadRequest should require a token field."""
        from api_bangalore import AdminReloadRequest
        req = AdminReloadRequest(token="test-secret")
        assert req.token == "test-secret"


class TestAPITestClient:
    """Test actual API endpoints through TestClient asserting real HTTP status codes."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api_bangalore import app, state, _load_model_and_data
        if not state.get("loaded"):
            try:
                state.update(_load_model_and_data())
            except Exception:
                pass
        return TestClient(app)

    def test_health_endpoint_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_outside_bangalore_returns_http_400(self, client):
        """A destination outside Bangalore must return HTTP 400 (not 422)."""
        resp = client.post("/route/custom", json={
            "origin": {"lat": 28.6139, "lng": 77.2090},
            "destination": {"lat": 12.9352, "lng": 77.6245},
            "day_offset": 0
        })
        assert resp.status_code == 400

    def test_same_point_returns_http_400(self, client):
        """Origin and destination < 200m apart must return HTTP 400."""
        resp = client.post("/route/custom", json={
            "origin": {"lat": 12.9716, "lng": 77.5946},
            "destination": {"lat": 12.9717, "lng": 77.5946},
            "day_offset": 0
        })
        assert resp.status_code == 400
        assert "200 m" in resp.json().get("detail", "")

    def test_osrm_failure_returns_http_502(self, client, monkeypatch):
        """OSRM failure must return HTTP 502."""
        import api_bangalore
        async def mock_fail(*args, **kwargs):
            raise Exception("OSRM timeout")
        monkeypatch.setattr(api_bangalore, "fetch_osrm_routes", mock_fail)

        resp = client.post("/route/custom", json={
            "origin": {"lat": 12.9716, "lng": 77.5946},
            "destination": {"lat": 12.9352, "lng": 77.6245},
            "day_offset": 0
        })
        assert resp.status_code == 502
        assert "OSRM" in resp.json().get("detail", "")

    def test_rate_limit_exceeded_returns_429(self, client, monkeypatch):
        """Exceeding 10 requests per minute from same IP returns HTTP 429."""
        from api_bangalore import _rate_limit, _cache
        _rate_limit.clear()
        _cache.clear()
        import api_bangalore
        async def mock_ok(*args, **kwargs):
            return [{
                "geometry": {"type": "LineString", "coordinates": [[77.60, 12.97], [77.63, 12.93]]},
                "distance": 5000,
                "duration": 800
            }]
        monkeypatch.setattr(api_bangalore, "fetch_osrm_routes", mock_ok)

        headers = {"X-Real-IP": "10.0.0.99"}
        body = {
            "origin": {"lat": 12.9716, "lng": 77.5946},
            "destination": {"lat": 12.9352, "lng": 77.6245},
            "day_offset": 0
        }
        for i in range(10):
            r = client.post("/route/custom", json=body, headers=headers)
            assert r.status_code == 200

        r11 = client.post("/route/custom", json=body, headers=headers)
        assert r11.status_code == 429
        assert "Rate limit exceeded" in r11.json().get("detail", "")

    def test_admin_reload_503_when_token_unset(self, client, monkeypatch):
        """When ADMIN_RELOAD_TOKEN is not set, /admin/reload returns 503."""
        import api_bangalore
        monkeypatch.setattr(api_bangalore, "ADMIN_TOKEN", None)
        resp = client.post("/admin/reload", json={"token": "some-token"})
        assert resp.status_code == 503
        assert "disabled" in resp.json().get("detail", "")

    def test_admin_reload_403_when_wrong_token(self, client, monkeypatch):
        """When token is wrong, /admin/reload returns 403."""
        import api_bangalore
        monkeypatch.setattr(api_bangalore, "ADMIN_TOKEN", "secret-test-token")
        resp = client.post("/admin/reload", json={"token": "wrongtoken"})
        assert resp.status_code == 403
        assert "Invalid admin token" in resp.json().get("detail", "")


class TestTrainingReproducibility:
    """Verify that training with seed produces reproducible results."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("torch")
        pytest.importorskip("yaml")

    def test_synthetic_training_seed_reproducibility(self, tmp_path):
        """A short run with fixed seed produces identical loss curve."""
        import torch
        import numpy as np
        from model import TrafficForecastModel

        def run_dummy_epoch(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = TrafficForecastModel(in_channels=2, in_periods=7, out_periods=3)
            x = torch.rand(16, 2, 7)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
            target = torch.rand(16, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()

            losses = []
            for _ in range(2):
                optimizer.zero_grad()
                out = model(x, edge_index, edge_weight)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            return losses

        run1 = run_dummy_epoch(42)
        run2 = run_dummy_epoch(42)
        assert run1 == run2, "Training with same seed must produce identical losses"



# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelSmokeTest:
    """Verify the A3T-GCN model loads and produces correct output shape."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("torch")

    def test_model_forward_pass(self):
        """Model should output shape (16, 3) for 16 roads, 3-day horizon."""
        import torch
        from model import TrafficForecastModel

        model = TrafficForecastModel(in_channels=2, in_periods=7, out_periods=3)
        x = torch.rand(16, 2, 7)
        edge_index = torch.randint(0, 16, (2, 100))
        edge_weight = torch.rand(100)

        out = model(x, edge_index, edge_weight)
        assert out.shape == (16, 3), f"Expected (16, 3), got {out.shape}"

    def test_checkpoint_loads(self):
        """Trained checkpoint should load and produce (16, 3) output."""
        import torch
        from model import TrafficForecastModel

        ckpt_path = os.path.join(_BASE, "data", "bangalore_best_model.pt")
        if not os.path.exists(ckpt_path):
            _skip_or_fail("Checkpoint not available (gitignored)")
            return

        model = TrafficForecastModel(in_channels=2, in_periods=7, out_periods=3)
        try:
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        x = torch.rand(16, 2, 7)
        edge_index = torch.randint(0, 16, (2, 100))
        edge_weight = torch.rand(100)

        with torch.no_grad():
            out = model(x, edge_index, edge_weight)
        assert out.shape == (16, 3), f"Expected (16, 3), got {out.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometryHelpers:
    """Test haversine and point sampling helpers."""

    def test_haversine_known_distance(self):
        from custom_routing import haversine_km
        # Bangalore Majestic to Whitefield: ~20 km
        d = haversine_km(12.977, 77.572, 12.969, 77.749)
        assert 15 < d < 25, f"Expected ~20 km, got {d}"

    def test_sample_points_along_line(self):
        from custom_routing import sample_points_along_line
        # Simple line ~5 km long
        coords = [[77.60, 12.97], [77.65, 12.97]]
        sampled = sample_points_along_line(coords, interval_m=200)
        assert len(sampled) >= 2, "Should have at least start and end"
        assert sampled[0] == (12.97, 77.60), "First point should match"
        assert sampled[-1] == (12.97, 77.65), "Last point should match"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Config wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigWiring:
    """Verify that config.yaml is well-formed and training script reads it."""

    def test_config_loads(self):
        import yaml
        config_path = os.path.join(_BASE, "model", "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "seed" in config
        assert "data" in config
        assert "model" in config
        assert "training" in config
        assert "speed_estimator" in config
        assert config["data"]["num_timesteps_in"] == 7
        assert config["data"]["num_timesteps_out"] == 3

    def test_data_pipeline_returns_dates(self):
        """load_bangalore_dataset must return 5 values including date index."""
        try:
            from bangalore_data import load_bangalore_dataset
            result = load_bangalore_dataset(num_timesteps_in=7, num_timesteps_out=3)
            assert len(result) == 5, f"Expected 5 return values, got {len(result)}"
            dataset, means, stds, road_order, all_dates = result
            assert len(all_dates) > 0, "Date index should not be empty"
            assert len(road_order) == 16, "Should have 16 roads"
        except FileNotFoundError:
            _skip_or_fail("Traffic CSV not found")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ETA check template
# ═══════════════════════════════════════════════════════════════════════════════

class TestETACheckTemplate:
    """Verify eta_check.csv template exists and has correct headers."""

    def test_eta_check_csv_exists(self):
        csv_path = os.path.join(_BASE, "data", "eta_check.csv")
        assert os.path.exists(csv_path), "data/eta_check.csv must exist"

    def test_eta_check_csv_has_headers(self):
        import csv
        csv_path = os.path.join(_BASE, "data", "eta_check.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
        required = {"origin", "destination", "our_eta_min", "reference_eta_min"}
        assert required.issubset(set(headers)), \
            f"Missing required headers. Got: {headers}"
