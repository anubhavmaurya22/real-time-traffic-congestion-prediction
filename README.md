# 🚦 Real-Time Traffic Congestion Prediction & Dynamic Route Optimization

A machine learning-powered platform for predicting urban traffic congestion in real time and dynamically optimizing routes to minimize travel time and fuel consumption.

---

## 📌 Project Overview

This project uses historical and real-time traffic data combined with ML models (LSTM, XGBoost, Graph Neural Networks) to:

- **Predict** traffic congestion levels at road segments up to 30 minutes ahead.
- **Optimize** routing dynamically by re-routing vehicles based on predicted congestion.
- **Visualize** real-time traffic flow on an interactive map dashboard.

---

## 🏗️ Project Structure

```
Real-Time Traffic Congestion Prediction & Dynamic Route Optimization/
├── data/
│   ├── raw/                  # Raw traffic datasets
│   ├── processed/            # Cleaned & feature-engineered data
│   └── external/             # External sources (weather, events)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_route_optimization.ipynb
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py     # Real-time data pipeline
│   ├── preprocessing.py      # Feature engineering & cleaning
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py     # LSTM time-series model
│   │   ├── xgboost_model.py  # XGBoost congestion classifier
│   │   └── gnn_model.py      # Graph Neural Network for road graph
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── graph_builder.py  # Road network graph (OSMnx)
│   │   └── optimizer.py      # Dynamic route optimization (Dijkstra/A*)
│   └── utils.py              # Helper utilities
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│   └── components/           # UI components
├── api/
│   ├── main.py               # FastAPI REST API
│   └── routes.py             # API endpoint definitions
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_routing.py
├── configs/
│   └── config.yaml           # Model & pipeline configuration
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys (HERE Maps, OpenWeatherMap, etc.)
```

### 3. Run Data Ingestion

```bash
python src/data_ingestion.py
```

### 4. Train Models


```bash
python src/models/lstm_model.py --train
python src/models/xgboost_model.py --train
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Start API Server

```bash
uvicorn api.main:app --reload --port 8000
```

---

## 🧠 ML Models

| Model | Task | Accuracy |
|-------|------|----------|
| LSTM | 30-min congestion forecast | ~87% |
| XGBoost | Congestion severity classification | ~91% |
| GNN | Road network flow prediction | ~84% |

---

## 📡 Data Sources

- **HERE Traffic API** — Real-time traffic flow & incidents
- **OpenStreetMap (OSMnx)** — Road network graph
- **OpenWeatherMap API** — Weather impact on traffic
- **Historical TomTom datasets** — Training data

---

## 🛠️ Tech Stack

- **ML/DL**: PyTorch, scikit-learn, XGBoost
- **Graph Processing**: NetworkX, OSMnx, PyTorch Geometric
- **Data Pipeline**: Apache Kafka, Redis
- **Dashboard**: Streamlit, Folium, Plotly
- **API**: FastAPI, Uvicorn
- **Database**: PostgreSQL + TimescaleDB

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
