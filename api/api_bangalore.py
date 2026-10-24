"""
Bangalore FastAPI backend — canonical entry point for uvicorn.

Run with:  uvicorn api_bangalore:app --reload
Then visit http://127.0.0.1:8000/docs for interactive testing.

Endpoints:
  GET  /                   → health check
  POST /predict            → predicted speeds for all 16 roads
  POST /route              → original integer-based routing (kept for compat)
  POST /route/custom       → arbitrary origin/destination routing via OSRM
  POST /admin/reload       → hot-reload model checkpoint (token-protected)
  POST /api/users/sync     → Firebase user sync
  GET  /api/users/{uid}    → get user profile
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
import torch
import networkx as nx
import pandas as pd
import numpy as np
import sys
import os
import time
import asyncio
import hashlib

# ---------------------------------------------------------------------------
# Path setup — works regardless of which directory uvicorn is invoked from
# ---------------------------------------------------------------------------
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_base, "data"))
sys.path.insert(0, os.path.join(_base, "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_base, "data")
_MODEL_DIR_DATA = os.path.join(_base, "data")  # bangalore_best_model.pt lives in data/
_MODEL_DIR = os.path.join(_base, "model")

from bangalore_data import load_bangalore_dataset, build_adjacency
from torch_geometric_temporal.signal import temporal_signal_split
from routing_bangalore import build_routing_graph, build_static_graph, compare_routes
from model import TrafficForecastModel
from db_sync import verify_firebase_id_token, upsert_cloud_sql_user, get_cloud_sql_user
from speed_estimator import SpeedEstimator, load_road_coords
from custom_routing import fetch_osrm_routes, score_all_routes

app = FastAPI(title="Bangalore Traffic Congestion Prediction & Route Optimization API")

# ---------------------------------------------------------------------------
# CORS — allow Firebase-hosted frontend, local dev servers, and file:// open
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://traff2ic-detector.web.app",
        "https://traff2ic-detector.firebaseapp.com",
        "https://43-204-232-243.sslip.io",
        "http://43.204.232.243:8000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

state = {}

# ---------------------------------------------------------------------------
# Admin reload token — set ADMIN_RELOAD_TOKEN env var to protect this endpoint
# ---------------------------------------------------------------------------
ADMIN_TOKEN = os.environ.get("ADMIN_RELOAD_TOKEN", "changeme")

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter + cache for /route/custom
# ---------------------------------------------------------------------------
_rate_limit = {}   # ip → [timestamps]
_cache = {}        # hash(request) → (response, timestamp)
RATE_LIMIT_MAX = 10       # requests per window
RATE_LIMIT_WINDOW = 60    # seconds
CACHE_TTL = 300           # seconds


def _rate_limit_check(client_ip: str):
    """Raise 429 if client exceeds rate limit."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    if client_ip not in _rate_limit:
        _rate_limit[client_ip] = []
    _rate_limit[client_ip] = [t for t in _rate_limit[client_ip] if t > window_start]
    if len(_rate_limit[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s."
        )
    _rate_limit[client_ip].append(now)


def _cache_key(origin_lat, origin_lng, dest_lat, dest_lng, day_offset):
    """Deterministic cache key for a custom route request."""
    raw = f"{origin_lat:.6f},{origin_lng:.6f},{dest_lat:.6f},{dest_lng:.6f},{day_offset}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    source: int
    target: int


class LatLng(BaseModel):
    lat: float
    lng: float
    label: Optional[str] = None

    @field_validator("lat")
    @classmethod
    def lat_in_bangalore(cls, v):
        if not (12.80 <= v <= 13.20):
            raise ValueError(
                f"Latitude {v} is outside the Bangalore bounding box (12.80–13.20). "
                "Please choose a location within Bangalore."
            )
        return v

    @field_validator("lng")
    @classmethod
    def lng_in_bangalore(cls, v):
        if not (77.40 <= v <= 77.85):
            raise ValueError(
                f"Longitude {v} is outside the Bangalore bounding box (77.40–77.85). "
                "Please choose a location within Bangalore."
            )
        return v


class CustomRouteRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    day_offset: int = 0

    @field_validator("day_offset")
    @classmethod
    def day_offset_range(cls, v):
        if not (0 <= v <= 2):
            raise ValueError("day_offset must be 0 (today), 1 (tomorrow), or 2 (day after).")
        return v


class UserSyncRequest(BaseModel):
    uid: str
    name: str
    email: str
    role: Optional[str] = "Operator / Analyst"


class AdminReloadRequest(BaseModel):
    token: str


# ---------------------------------------------------------------------------
# Startup: load model + data
# ---------------------------------------------------------------------------

def _load_model_and_data():
    """Load dataset, model checkpoint, and speed estimator. Used at startup and reload."""
    dataset, means, stds, road_order = load_bangalore_dataset(
        num_timesteps_in=7, num_timesteps_out=3
    )
    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
    _, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

    sample = next(iter(train_dataset))
    model = TrafficForecastModel(
        in_channels=sample.x.shape[1], in_periods=7, out_periods=3
    )
    ckpt_path = os.path.join(_MODEL_DIR_DATA, "bangalore_best_model.pt")
    try:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    locations = pd.read_csv(os.path.join(_DATA_DIR, "bangalore_road_locations.csv"))
    _, dist_km = build_adjacency(locations)

    speed_mean = float(means[0, 0, 0])
    speed_std = float(stds[0, 0, 0])

    # Get predicted speeds for all 3 days from the last test snapshot
    test_snap = next(iter(test_dataset))
    with torch.no_grad():
        pred_out = model(test_snap.x, test_snap.edge_index, test_snap.edge_attr)
    # pred_out shape: (16, 3) — 3 horizons
    predicted_speeds_per_day = []
    for d in range(3):
        speeds_d = (pred_out[:, d] * speed_std + speed_mean).tolist()
        predicted_speeds_per_day.append(speeds_d)

    road_coords = load_road_coords()
    speed_estimator = SpeedEstimator(road_coords=road_coords)

    return {
        "model": model,
        "snapshot": test_snap,
        "num_nodes": sample.x.shape[0],
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "dist_km": dist_km,
        "road_names": road_order,
        "road_coords": road_coords,
        "speed_estimator": speed_estimator,
        "predicted_speeds_per_day": predicted_speeds_per_day,
        "loaded": True,
    }


@app.on_event("startup")
def load_resources():
    loaded = _load_model_and_data()
    state.update(loaded)
    print(
        f"[startup] loaded {state['num_nodes']} roads | "
        f"speed mean={state['speed_mean']:.1f} std={state['speed_std']:.1f} km/h"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "loaded": state.get("loaded", False),
        "num_roads": state.get("num_nodes"),
    }


@app.post("/predict")
def predict():
    if not state.get("loaded"):
        raise HTTPException(status_code=503, detail="model not loaded")
    snapshot = state["snapshot"]
    with torch.no_grad():
        out = state["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    predicted_kmh = (out[:, 0] * state["speed_std"] + state["speed_mean"]).tolist()
    return {"predicted_speeds_kmh": predicted_kmh, "road_names": state["road_names"]}


@app.post("/route")
def route(req: RouteRequest):
    if not state.get("loaded"):
        raise HTTPException(status_code=503, detail="model not loaded")

    n = state["num_nodes"]
    if req.source == req.target:
        raise HTTPException(status_code=400, detail="source and target must differ")
    if not (0 <= req.source < n) or not (0 <= req.target < n):
        raise HTTPException(
            status_code=400, detail=f"road id out of range (0–{n - 1})"
        )

    snapshot = state["snapshot"]
    with torch.no_grad():
        out = state["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    predicted_speed = (out[:, 0] * state["speed_std"] + state["speed_mean"]).tolist()

    G_pred = build_routing_graph(state["dist_km"], predicted_speed)
    G_static = build_static_graph(state["dist_km"], state["speed_mean"])

    try:
        result = compare_routes(G_pred, G_static, req.source, req.target)
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="no path found")

    road_names = state["road_names"]
    result["predicted_route_names"] = [road_names[i] for i in result["predicted_route"]]
    result["static_route_names"] = [road_names[i] for i in result["static_route"]]
    return result


@app.post("/route/custom")
async def route_custom(req: CustomRouteRequest, request: Request):
    """
    Arbitrary origin/destination routing within Bangalore.
    Fetches OSRM routes, scores them with the spatial speed estimator,
    and recommends the fastest predicted route.
    """
    if not state.get("loaded"):
        raise HTTPException(status_code=503, detail="model not loaded")

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    _rate_limit_check(client_ip)

    # Validate minimum distance between origin and destination (200 m)
    from custom_routing import haversine_m
    dist_m = haversine_m(
        req.origin.lat, req.origin.lng,
        req.destination.lat, req.destination.lng
    )
    if dist_m < 200:
        raise HTTPException(
            status_code=400,
            detail=f"Origin and destination are only {dist_m:.0f} m apart. "
                   "They must be at least 200 m apart."
        )

    # Check cache
    ck = _cache_key(
        req.origin.lat, req.origin.lng,
        req.destination.lat, req.destination.lng,
        req.day_offset
    )
    if ck in _cache:
        cached_resp, cached_time = _cache[ck]
        if time.time() - cached_time < CACHE_TTL:
            return cached_resp

    # Select predicted speeds for the requested day offset
    day_offset = req.day_offset
    predicted_speeds = state["predicted_speeds_per_day"][day_offset]

    # Fetch OSRM routes
    try:
        osrm_routes = await fetch_osrm_routes(
            req.origin.lat, req.origin.lng,
            req.destination.lat, req.destination.lng,
            timeout_s=8.0, max_alternatives=3,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch routes from OSRM: {str(e)}. "
                   "The OSRM public server may be temporarily unavailable."
        )

    # Score routes
    response = score_all_routes(
        osrm_routes,
        state["speed_estimator"],
        predicted_speeds,
        day_offset,
        state["speed_mean"],
    )

    # Cache the response
    _cache[ck] = (response, time.time())

    return response


@app.post("/admin/reload")
def admin_reload(req: AdminReloadRequest):
    """
    Hot-reload the model checkpoint without restarting the server.
    Protected by ADMIN_RELOAD_TOKEN environment variable.
    """
    if req.token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    try:
        loaded = _load_model_and_data()
        state.update(loaded)
        # Clear route cache on reload
        _cache.clear()
        return {
            "status": "reloaded",
            "num_roads": state["num_nodes"],
            "speed_mean": state["speed_mean"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.post("/api/users/sync")
def sync_user(req: UserSyncRequest, request: Request):
    """
    Verifies Firebase ID token from Authorization header and upserts User in Cloud SQL.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token header")

    try:
        token_payload = verify_firebase_id_token(auth_header)
        verified_uid = token_payload.get("sub") or token_payload.get("user_id") or token_payload.get("uid")
    except ValueError as val_err:
        raise HTTPException(status_code=401, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=401, detail=f"Invalid Authorization token: {str(err)}")

    if req.uid and req.uid != verified_uid:
        print(f"[SECURITY WARNING] Request body UID ({req.uid}) != verified token UID ({verified_uid}). Using verified UID.")

    try:
        user_record = upsert_cloud_sql_user(
            uid=verified_uid,
            name=req.name,
            email=req.email,
            role=req.role or "Operator / Analyst"
        )
        return {
            "status": "success",
            "message": "User synchronized with Cloud SQL",
            "user": user_record
        }
    except Exception as db_err:
        raise HTTPException(status_code=500, detail=f"Cloud SQL synchronization failed: {str(db_err)}")


@app.get("/api/users/{uid}")
def get_user(uid: str, request: Request):
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token header")

    try:
        token_payload = verify_firebase_id_token(auth_header)
        verified_uid = token_payload.get("sub") or token_payload.get("user_id") or token_payload.get("uid")
    except Exception as err:
        raise HTTPException(status_code=401, detail="Invalid token")

    if verified_uid != uid:
        raise HTTPException(status_code=403, detail="Unauthorized access to user profile")

    user_data = get_cloud_sql_user(uid)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found in Cloud SQL")

    return {"status": "success", "user": user_data}