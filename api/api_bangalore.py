"""
Bangalore FastAPI backend — canonical entry point for uvicorn.

Run with:  uvicorn api_bangalore:app --reload
Then visit http://127.0.0.1:8000/docs for interactive testing.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import networkx as nx
import pandas as pd
import sys
import os

# ---------------------------------------------------------------------------
# Path setup — works regardless of which directory uvicorn is invoked from
# ---------------------------------------------------------------------------
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_base, "data"))
sys.path.insert(0, os.path.join(_base, "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_base, "data")
_MODEL_DIR = os.path.join(_base, "data")  # bangalore_best_model.pt lives in data/

from bangalore_data import load_bangalore_dataset, build_adjacency
from torch_geometric_temporal.signal import temporal_signal_split
from routing_bangalore import build_routing_graph, build_static_graph, compare_routes
from model import TrafficForecastModel
from db_sync import verify_firebase_id_token, upsert_cloud_sql_user, get_cloud_sql_user

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


class RouteRequest(BaseModel):
    source: int
    target: int


class UserSyncRequest(BaseModel):
    uid: str
    name: str
    email: str
    role: Optional[str] = "Operator / Analyst"


@app.on_event("startup")
def load_resources():
    dataset, means, stds, road_order = load_bangalore_dataset(
        num_timesteps_in=7, num_timesteps_out=3
    )
    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
    _, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

    sample = next(iter(train_dataset))
    model = TrafficForecastModel(
        in_channels=sample.x.shape[1], in_periods=7, out_periods=3
    )
    try:
        model.load_state_dict(
            torch.load(
                os.path.join(_MODEL_DIR, "bangalore_best_model.pt"),
                weights_only=True,
            )
        )
    except TypeError:
        model.load_state_dict(
            torch.load(os.path.join(_MODEL_DIR, "bangalore_best_model.pt"))
        )
    model.eval()

    locations = pd.read_csv(os.path.join(_DATA_DIR, "bangalore_road_locations.csv"))
    _, dist_km = build_adjacency(locations)

    state["model"] = model
    state["snapshot"] = next(iter(test_dataset))
    state["num_nodes"] = sample.x.shape[0]
    state["speed_mean"] = float(means[0, 0, 0])
    state["speed_std"] = float(stds[0, 0, 0])
    state["dist_km"] = dist_km
    state["road_names"] = road_order
    state["loaded"] = True
    print(
        f"[startup] loaded {state['num_nodes']} roads | "
        f"speed mean={state['speed_mean']:.1f} std={state['speed_std']:.1f} km/h"
    )


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