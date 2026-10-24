# Real-Time Traffic Congestion Prediction & Route Optimization

**StreetFlow Live** — AI-powered traffic forecasting and dynamic route
optimization for Bangalore, using an A3T-GCN (Attention Temporal Graph
Convolutional Network) on a 16-road sensor network.

## Architecture

```
frontend/         Leaflet dashboard (Firebase Hosting)
api/              FastAPI backend (AWS EC2 + nginx)
model/            A3T-GCN model, speed estimator, evaluation
data/             Bangalore dataset, routing, custom routing
tests/            pytest suite
```

## Setup

```bash
pip install torch
python -c "import torch; print(torch.__version__)"   # note the version

pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-<VERSION>.html

pip install -r requirements.txt
```

## Training

```bash
cd data
python train_bangalore.py
```

This trains the A3T-GCN model on the Bangalore traffic dataset (16 roads,
7-day input → 3-day forecast) and saves `bangalore_best_model.pt`.

Hyperparameters are in [`model/config.yaml`](model/config.yaml).

## Evaluation

```bash
python model/evaluate_bangalore.py
```

Produces:
- `model/results.json` — full metrics (MAE, RMSE, MAPE per horizon, per
  method, per road, spatial estimator LOO CV)
- `model/error_vs_horizon.png` — forecast error across 3-day horizon
- `model/predicted_vs_observed.png` — best and worst road time series
- `model/training_curve.png` — if training log CSV exists

**Evaluation compares:**
| Method | What it does |
|---|---|
| A3T-GCN | Trained model predictions |
| Persistence | Repeat the last observed day |
| Historical Average | Training-set mean per road per horizon |

**Spatial estimator** (leave-one-road-out CV):
| Method | What it does |
|---|---|
| IDW | Inverse-distance weighted average of k nearest road speeds |
| Learned (GBR) | Gradient-boosting regressor on (lat, lng, distances, speeds) |
| Global Mean | City-wide average speed baseline |

## Running the API

```bash
cd api
uvicorn api_bangalore:app --reload
# Visit http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predicted speeds for all 16 roads |
| `POST` | `/route` | Original integer-based routing (road IDs 0–15) |
| `POST` | `/route/custom` | **New** — arbitrary origin/destination routing |
| `POST` | `/admin/reload` | Hot-reload model checkpoint (token-protected) |
| `POST` | `/api/users/sync` | Firebase user sync |
| `GET` | `/api/users/{uid}` | Get user profile |

### POST /route/custom — Example

**Request:**
```json
{
  "origin": {"lat": 12.9716, "lng": 77.5946, "label": "Indiranagar Metro"},
  "destination": {"lat": 12.8440, "lng": 77.6610, "label": "Electronic City"},
  "day_offset": 0
}
```

**Response:**
```json
{
  "routes": [
    {
      "id": "route_0",
      "geometry": {"type": "LineString", "coordinates": [[77.59, 12.97], ...]},
      "distance_km": 18.4,
      "eta_pred_min": 32.5,
      "eta_static_min": 35.1,
      "avg_speed_kmh": 33.9,
      "recommended": true
    },
    {
      "id": "route_1",
      "geometry": {"type": "LineString", "coordinates": [...]},
      "distance_km": 20.1,
      "eta_pred_min": 36.8,
      "eta_static_min": 38.2,
      "avg_speed_kmh": 32.8,
      "recommended": false
    }
  ],
  "time_saved_min": 2.6,
  "improvement_pct": 7.4,
  "coverage_warning": false
}
```

**Validation:**
- Coordinates must be inside the Bangalore bounding box (lat 12.80–13.20, lng 77.40–77.85)
- Origin and destination must be ≥ 200 m apart
- `day_offset` must be 0 (today), 1 (tomorrow), or 2 (day after)

**Error responses:**
- `400` — invalid coordinates (outside Bangalore) or same-point
- `502` — OSRM routing server unreachable
- `429` — rate limit exceeded (10 req/min per IP)

### POST /admin/reload

Hot-reload the model checkpoint without restarting the server. Set the
`ADMIN_RELOAD_TOKEN` environment variable and pass it in the request body:

```bash
curl -X POST http://localhost:8000/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"token": "your-secret-token"}'
```

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Speed estimator: near/far from roads, coverage warnings, LOO CV metrics
- Bounding-box validation and same-point rejection
- Route scoring with mocked OSRM responses
- API error handling (400, 502)
- Model smoke test: forward pass shape (16, 3), checkpoint loading

## Frontend

The dashboard at `frontend/dashboard.html` provides:
- **Free-text search** — type any place in Bangalore (powered by Nominatim
  geocoding, debounced, rate-limited per their policy)
- **"Use my location"** — GPS origin with bounding-box validation
- **"Pick on map"** — click to set origin or destination
- **Swap button** — exchange origin and destination
- **Day selector** — today, tomorrow, day after (maps to model's 3-day horizon)
- **Route display** — recommended route in solid gold, alternatives in dashed grey
- **ETA and time-saved** display with coverage warning when applicable
- Sensor overlay with speed color coding (auto-refreshes every 30 s)

## Limitations

- **Daily granularity** — the model forecasts daily average speeds, not
  5-minute or hourly. ETAs are approximate day-level estimates.
- **16 approximated road coordinates** — 14 of 16 use parent-area centroids,
  not exact junction positions. The spatial speed estimator interpolates
  between these, which adds uncertainty.
- **OSRM public server** — the demo uses OSRM's public routing service, which
  has usage limits and is not suitable for production traffic. For deployment,
  use a self-hosted OSRM instance or a paid service (Mapbox, Google).
- **No published benchmark** — this is a new baseline on this dataset; results
  are not comparable to published traffic forecasting numbers on METR-LA/PeMS.

## Project structure

```
data/
  bangalore_data.py          Dataset pipeline (CSV → temporal signal)
  bangalore_road_locations.csv  16 road coordinates
  routing_bangalore.py       Original integer-based routing (networkx)
  custom_routing.py          NEW: OSRM client + route scoring
  train_bangalore.py         Training script

model/
  model.py                   A3T-GCN model (TrafficForecastModel)
  config.yaml                NEW: centralized hyperparameters
  speed_estimator.py         NEW: spatial speed estimation (IDW/GBR)
  evaluate_bangalore.py      NEW: full evaluation + plots
  results.json               Generated by evaluation script
  *.png                      Generated plots

api/
  api_bangalore.py           FastAPI backend (all endpoints)
  db_sync.py                 Firebase user sync
  Dockerfile                 Container build

frontend/
  dashboard.html             Main dashboard (Leaflet + routing UI)
  api-config.js              API base URL config
  firebase-init.js           Firebase Auth init
  theme.css                  Shared design tokens

tests/
  test_streetflow.py         pytest suite
```