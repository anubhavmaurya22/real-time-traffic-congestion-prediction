"""
Week 4, Days 25-27: FastAPI backend.

Verified before delivery: the startup-loading pattern (load model once at
app startup, not per-request) and input validation (out-of-range sensor
IDs, source==target) were tested with FastAPI's TestClient and confirmed
working. The model/routing logic itself is your already-verified model.py
and routing.py from Weeks 2-4 -- this file just wires them into HTTP
endpoints.

Run with:  uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API testing.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import networkx as nx
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import sys
sys.path.append("../model")
from model import TrafficForecastModel
from routing import build_routing_graph, build_static_graph, compare_routes

app = FastAPI(title="Traffic Congestion Prediction & Route Optimization API")

SPEED_MEAN, SPEED_STD = 53.6, 20.16  # from your normalize_data.py -- confirm these match

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


@app.on_event("startup")
def load_resources():
    """Loads model, dataset, and current snapshot once -- not per-request.
    A real deployment would refresh the snapshot periodically (e.g. every
    5 minutes) rather than using one fixed test snapshot forever."""
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
    _, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

    sample = next(iter(train_dataset))
    model = TrafficForecastModel(in_channels=sample.x.shape[1], periods=12)
    model.load_state_dict(torch.load("../model/best_model.pt"))
    model.eval()

    snapshot = next(iter(test_dataset))

    state["model"] = model
    state["snapshot"] = snapshot
    state["num_nodes"] = snapshot.x.shape[0]
    state["loaded"] = True


@app.get("/")
def root():
    return {"status": "ok", "loaded": state.get("loaded", False), "num_sensors": state.get("num_nodes")}


@app.post("/predict")
def predict():
    if not state.get("loaded"):
        raise HTTPException(status_code=503, detail="model not loaded")

    snapshot = state["snapshot"]
    with torch.no_grad():
        out = state["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    predicted_mph = (out[:, 0] * SPEED_STD + SPEED_MEAN).tolist()
    return {"predicted_speeds_mph": predicted_mph}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if not state.get("loaded"):
        raise HTTPException(status_code=503, detail="model not loaded")

    num_nodes = state["num_nodes"]
    if req.source == req.target:
        raise HTTPException(status_code=400, detail="source and target must differ")
    if not (0 <= req.source < num_nodes) or not (0 <= req.target < num_nodes):
        raise HTTPException(status_code=400, detail=f"sensor id out of range (0-{num_nodes-1})")

    snapshot = state["snapshot"]
    with torch.no_grad():
        out = state["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    predicted_speed = out[:, 0] * SPEED_STD + SPEED_MEAN

    G_pred = build_routing_graph(snapshot.edge_index, snapshot.edge_attr, predicted_speed)
    G_static = build_static_graph(snapshot.edge_index, snapshot.edge_attr)

    try:
        result = compare_routes(G_pred, G_static, req.source, req.target)
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="no path found between these sensors")

    return RouteResponse(
        source=req.source,
        target=req.target,
        predicted_route=result["predicted_route"],
        static_route=result["static_route"],
        improvement_pct=result["improvement_pct"],
    )