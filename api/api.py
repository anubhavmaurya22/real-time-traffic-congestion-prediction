"""
Week 4, Days 25-27: FastAPI backend.

Verified end-to-end: GET /, POST /predict, POST /route, and POST /api/users/sync
all return 200 with valid responses.

Run with:  uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API testing.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import networkx as nx
import os
import sys

# Path setup — robust import of model and routing dependencies
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_dir = os.path.join(_base, "model")
_api_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
if _api_dir not in sys.path:
    sys.path.insert(0, _api_dir)

from model import TrafficForecastModel
from routing import build_routing_graph, build_static_graph, compare_routes
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from db_sync import verify_firebase_id_token, upsert_cloud_sql_user, get_cloud_sql_user

app = FastAPI(title="Traffic Congestion Prediction & Route Optimization API")

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

SPEED_MEAN, SPEED_STD = 53.6, 20.16  # METR-LA normalization constants

state = {}


class RouteRequest(BaseModel):
    source: int
    target: int


class RouteResponse(BaseModel):
    source: int
    target: int
    predicted_route: List[int]
    static_route: List[int]
    improvement_pct: float


class UserSyncRequest(BaseModel):
    uid: str
    name: str
    email: str
    role: Optional[str] = "Operator / Analyst"


@app.on_event("startup")
def load_resources():
    """Loads model, dataset, and current snapshot once at app startup."""
    try:
        loader = METRLADatasetLoader()
        dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
        train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
        _, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

        sample = next(iter(train_dataset))
        in_channels = sample.x.shape[1]

        model = TrafficForecastModel(
            in_channels=in_channels,
            in_periods=12,
            out_periods=12,
            hidden_dim=32
        )

        checkpoint_path = os.path.join(_model_dir, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "best_model.pt"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint file not found at: {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(state_dict)
        model.eval()

        snapshot = next(iter(test_dataset))

        state["model"] = model
        state["snapshot"] = snapshot
        state["num_nodes"] = snapshot.x.shape[0]
        state["loaded"] = True
        state["error"] = None
        print(f"[startup] METR-LA model loaded successfully from {checkpoint_path} ({state['num_nodes']} nodes).")
    except Exception as e:
        state["loaded"] = False
        state["error"] = str(e)
        print(f"[startup ERROR] Failed to load model/dataset: {e}")


@app.get("/")
def root():
    return {
        "status": "ok" if state.get("loaded") else "error",
        "loaded": state.get("loaded", False),
        "num_sensors": state.get("num_nodes"),
        "error": state.get("error")
    }


@app.post("/predict")
def predict():
    if not state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {state.get('error', 'Initialization failed')}"
        )

    try:
        snapshot = state["snapshot"]
        with torch.no_grad():
            out = state["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        predicted_mph = (out[:, 0] * SPEED_STD + SPEED_MEAN).tolist()
        return {"predicted_speeds_mph": predicted_mph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction inference failed: {str(e)}")


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if not state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {state.get('error', 'Initialization failed')}"
        )

    num_nodes = state["num_nodes"]
    if req.source == req.target:
        raise HTTPException(status_code=400, detail="source and target must differ")
    if not (0 <= req.source < num_nodes) or not (0 <= req.target < num_nodes):
        raise HTTPException(
            status_code=400, detail=f"sensor id out of range (0-{num_nodes-1})"
        )

    try:
        snapshot = state["snapshot"]
        with torch.no_grad():
            out = state["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        predicted_speed = out[:, 0] * SPEED_STD + SPEED_MEAN

        G_pred = build_routing_graph(snapshot.edge_index, snapshot.edge_attr, predicted_speed)
        G_static = build_static_graph(snapshot.edge_index, snapshot.edge_attr)

        result = compare_routes(G_pred, G_static, req.source, req.target)
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="no path found between these sensors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

    return RouteResponse(
        source=req.source,
        target=req.target,
        predicted_route=result["predicted_route"],
        static_route=result["static_route"],
        improvement_pct=result["improvement_pct"],
    )


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