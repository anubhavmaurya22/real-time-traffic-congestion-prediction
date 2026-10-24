# Real-Time Traffic Congestion Prediction & Route Optimization

**StreetFlow Live** — AI-powered traffic forecasting and dynamic route
optimization for Bangalore, using an A3T-GCN (Attention Temporal Graph
Convolutional Network) on a 16-road sensor network.

> **Dataset Notice**: The Bangalore dataset is historical (spanning 952 days).
> The live demo forecasts the 3 days following the last recorded date in the
> dataset (`data_end_date`), demonstrating real-time model inference and routing
> mechanics rather than live telemetry.

---

## Architecture

```
frontend/         Leaflet dashboard (Firebase Hosting)
api/              FastAPI backend (AWS EC2 + nginx reverse proxy)
model/            A3T-GCN model, speed estimator, evaluation
data/             Bangalore dataset, routing, custom routing
future_work/      Multi-city data collection experiments (Lucknow/Kanpur)
tests/            pytest test suite (41 tests)
```

---

## Setup & Installation

### Local Setup (Python 3.11 / 3.12)

```bash
# 1. Install PyTorch (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"

# 2. Install PyG prebuilt wheels (replace <VERSION> with output above, e.g. 2.12.0+cpu)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>.html

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
cd data
python train_bangalore.py
```

- Reads hyperparameters from [`model/config.yaml`](model/config.yaml) (epochs, learning rate, patience, seed).
- Sets reproducibility seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).
- Logs per-epoch train and validation loss to [`model/training_log.csv`](model/training_log.csv).
- Saves best checkpoint to `data/bangalore_best_model.pt`.

---

## Evaluation

```bash
python model/evaluate_bangalore.py
```

Executes single-command end-to-end evaluation:
1. **Multi-Horizon Forecast**: MAE, RMSE, MAPE per day horizon (Day 1, 2, 3) comparing:
   - **A3T-GCN** model
   - **Persistence** baseline (repeat last observed day)
   - **Historical Average** baseline
2. **Per-Road Analysis**: Identifies best- and worst-performing roads.
3. **Spatial Speed Estimator LOO CV**:
   - **Strict Leave-One-Out CV**: Excludes near-duplicate roads within 0.5 km (e.g. 100 Feet Rd / CMH Rd twins) to prevent spatial leakage.
   - **Original Leave-One-Out CV**: Single road exclusion.
   - Evaluated on both model-predicted speeds and observed ground truth.
   - Automatically selects best estimator (`idw` vs `learned`) and updates `model/config.yaml`.
4. **ETA Sanity Check**: Computes mean absolute difference from [`data/eta_check.csv`](data/eta_check.csv) (if user-provided).
5. **Generates Outputs**:
   - `model/results.json` — complete evaluation metrics
   - `model/error_vs_horizon.png` — error curve across forecast horizons
   - `model/predicted_vs_observed.png` — predicted vs observed speeds on best/worst roads
   - `model/training_curve.png` — training & validation loss curves

---

## Running the API Locally

```bash
cd api
uvicorn api_bangalore:app --reload
```
Interactive documentation available at `http://127.0.0.1:8000/docs`.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Basic status check |
| `GET` | `/health` | Health check returning model status, road count, `data_end_date`, and `forecast_dates` |
| `POST` | `/predict` | Predicted speeds for all 16 Bangalore roads + forecast dates |
| `POST` | `/route` | Integer-based node routing (0–15) comparing static vs predicted shortest path |
| `POST` | `/route/custom` | Arbitrary origin/destination routing with OSRM alternatives scored by spatial speed estimator |
| `POST` | `/admin/reload` | Hot-reload model checkpoint without server restart (token-protected) |
| `POST` | `/api/users/sync` | Firebase Auth user synchronisation |
| `GET` | `/api/users/{uid}` | Get user profile from database |

### POST /route/custom

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
      "geometry": {
        "type": "LineString",
        "coordinates": [[77.5946, 12.9716], [77.6200, 12.9300], [77.6610, 12.8440]]
      },
      "distance_km": 18.4,
      "eta_pred_min": 32.5,
      "eta_avg_speed_min": 35.1,
      "avg_speed_kmh": 33.9,
      "recommended": true,
      "coverage_warning": false
    },
    {
      "id": "route_1",
      "geometry": {
        "type": "LineString",
        "coordinates": [[77.5946, 12.9716], [77.6400, 12.9100], [77.6610, 12.8440]]
      },
      "distance_km": 20.1,
      "eta_pred_min": 36.8,
      "eta_avg_speed_min": 38.2,
      "avg_speed_kmh": 32.8,
      "recommended": false,
      "coverage_warning": false
    }
  ],
  "time_saved_min": 4.3,
  "improvement_pct": 11.7,
  "default_route_eta_min": 36.8,
  "coverage_warning": false,
  "data_end_date": "2024-03-31",
  "forecast_dates": ["2024-04-01", "2024-04-02", "2024-04-03"]
}
```

**Validation & Error Handling:**
- `400 Bad Request`: Coordinates outside Bangalore bounding box (lat 12.80–13.20, lng 77.40–77.85) or origin/destination < 200 m apart.
- `429 Too Many Requests`: Rate limit of 10 requests/minute per client IP (tracked via `X-Real-IP` from nginx).
- `502 Bad Gateway`: OSRM routing server unreachable or failed.
- `503 Service Unavailable`: If model data not yet loaded or `/admin/reload` called without `ADMIN_RELOAD_TOKEN` configured.

---

## Running Tests

Run the full automated test suite (41 tests):
```bash
pytest tests/ -v
```

To run in strict CI mode (fails loudly if optional packages are missing):
```bash
STRICT_TESTS=1 pytest tests/ -v
```

---

## Server Deployment (AWS EC2)

The backend is deployed on an AWS EC2 instance (Amazon Linux 2023 / Ubuntu) behind an Nginx reverse proxy.

### 1. Setting ADMIN_RELOAD_TOKEN
The admin reload token protects the `/admin/reload` endpoint. Store it in `/home/ec2-user/.streetflow_env`:
```bash
echo "ADMIN_RELOAD_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" > /home/ec2-user/.streetflow_env
chmod 600 /home/ec2-user/.streetflow_env
```
The systemd service file reads this file via `EnvironmentFile=/home/ec2-user/.streetflow_env`.

### 2. Copying the Model Checkpoint
Because `.pt` files are gitignored, copy your trained checkpoint to the server via SCP:
```bash
scp -i deploy/streetflow-key.pem data/bangalore_best_model.pt ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/streetflow/data/
```

### 3. Redeploy / Update
To pull updates and restart the service on EC2:
```bash
cd /home/ec2-user/streetflow
git pull origin main
python3.11 -m pip install -r api/requirements.txt
sudo systemctl restart streetflow

# Verify health
curl http://localhost:8000/health
```

---

## Limitations to State in Reports

1. **Daily Aggregation**: Predictions are daily averages, suitable for multi-day route planning rather than minute-by-minute congestion spikes.
2. **Approximate Centroid Coordinates**: 14 of 16 sensor roads represent area centroids rather than exact junction bounds; inverse-distance weighting interpolates between them.
3. **OSRM Public Gateway**: Public OSRM API has rate caps; production deployments should point to a dedicated private OSRM or Valhalla instance.
4. **Historical Evaluation**: Evaluated against historical Bangalore traffic dataset; no published peer benchmark existed for this specific 16-road sensor set.

---

## Project Structure

```
├── api/
│   ├── api_bangalore.py            # FastAPI backend application
│   ├── db_sync.py                  # Firebase token verification & DB sync
│   ├── requirements.txt            # Production runtime dependencies
│   └── Dockerfile                  # Container definition
├── data/
│   ├── bangalore_data.py           # Data loader & static temporal graph builder
│   ├── bangalore_road_locations.csv# Coordinates of 16 Bangalore monitored roads
│   ├── Banglore_traffic_Dataset.csv# Monitored daily speed & congestion history
│   ├── custom_routing.py           # OSRM async routing client & route scoring
│   ├── routing_bangalore.py        # Static graph baseline routing
│   ├── train_bangalore.py          # Seeded training script with loss logging
│   └── eta_check.csv               # Template for real-world ETA verification
├── model/
│   ├── model.py                    # A3T-GCN PyTorch Geometric Temporal network
│   ├── speed_estimator.py          # Spatial speed estimator (IDW / GBR) & strict LOO CV
│   ├── evaluate_bangalore.py       # Metrics evaluation & plot generator
│   ├── config.yaml                 # Centralized hyperparameters & estimator selection
│   ├── results.json                # Evaluated metrics report
│   ├── error_vs_horizon.png        # Generated plot: error vs forecast horizon
│   └── predicted_vs_observed.png   # Generated plot: predicted vs actual road speeds
├── frontend/
│   ├── dashboard.html              # Leaflet web app with Nominatim geocoding & routing UI
│   ├── api-config.js               # API URL environment switcher
│   ├── firebase-init.js            # Firebase authentication integration
│   └── theme.css                   # Responsive dark-mode stylesheet
├── future_work/
│   ├── README.md                   # Multi-city traffic expansion documentation
│   ├── collect_traffic.py          # TomTom Flow Segment API live collector
│   ├── lucknow_kanpur_traffic.csv  # Initial exploratory dataset
│   └── requirements.txt            # Collector requirements (requests)
├── deploy/
│   ├── nginx-streetflow.conf       # Nginx proxy config with X-Real-IP forwarding
│   └── setup_ec2_fixed.sh          # EC2 provisioning script with wheel dependencies
└── tests/
    └── test_streetflow.py          # Pytest suite with strict-mode support
```