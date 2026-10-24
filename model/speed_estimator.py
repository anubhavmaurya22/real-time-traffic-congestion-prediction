"""
Spatial speed estimator — predicts speed at arbitrary (lat, lng) points
using the 16 model-predicted road speeds.

Two methods:
  1. IDW (inverse-distance weighting): Gaussian-kernel weighted average of
     the k nearest predicted speeds, no training required.
  2. Learned: a gradient-boosting regressor taking (lat, lng, distances to
     k nearest roads, their predicted speeds, day_offset) → speed.

Evaluated via leave-one-road-out cross-validation (see evaluate_estimator.py).
"""

import os
import math
import json
import numpy as np
import pandas as pd
import joblib
import yaml

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Haversine ────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── Load road coordinates ────────────────────────────────────────────────────

def load_road_coords(csv_path=None):
    """Returns list of dicts: [{index, road_name, lat, lng}, ...]"""
    if csv_path is None:
        csv_path = os.path.join(_DATA_DIR, "bangalore_road_locations.csv")
    df = pd.read_csv(csv_path)
    roads = []
    for _, row in df.iterrows():
        roads.append({
            "index": int(row["index"]),
            "road_name": row["road_name"],
            "lat": float(row["latitude"]),
            "lng": float(row["longitude"]),
        })
    return roads


def load_config():
    """Load model/config.yaml."""
    config_path = os.path.join(_MODEL_DIR, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── IDW estimator ────────────────────────────────────────────────────────────

def idw_speed(lat, lng, road_coords, predicted_speeds, power=2.0, k=5,
              min_speed=5.0):
    """
    Inverse-distance weighted average of the k nearest road speeds.
    Uses a Gaussian-kernel weighting: w = 1 / (d^power + eps).
    """
    distances = []
    for road in road_coords:
        d = haversine_km(lat, lng, road["lat"], road["lng"])
        distances.append((d, road["index"]))
    distances.sort(key=lambda x: x[0])

    # Take k nearest
    nearest = distances[:k]
    eps = 1e-6  # avoid division by zero if query is exactly on a road

    weighted_sum = 0.0
    weight_total = 0.0
    for d, idx in nearest:
        w = 1.0 / (d ** power + eps)
        weighted_sum += w * predicted_speeds[idx]
        weight_total += w

    speed = weighted_sum / weight_total if weight_total > 0 else np.mean(predicted_speeds)
    return max(speed, min_speed)


# ── Learned estimator (gradient boosting) ─────────────────────────────────────

def build_features(lat, lng, road_coords, predicted_speeds, day_offset, k=5):
    """
    Build feature vector for the learned estimator:
    [lat, lng, d1..dk, speed1..speedk, day_offset]
    """
    distances = []
    for road in road_coords:
        d = haversine_km(lat, lng, road["lat"], road["lng"])
        distances.append((d, road["index"]))
    distances.sort(key=lambda x: x[0])

    nearest = distances[:k]
    feats = [lat, lng]
    for d, idx in nearest:
        feats.append(d)
    for d, idx in nearest:
        feats.append(predicted_speeds[idx])
    feats.append(float(day_offset))
    return np.array(feats, dtype=np.float64)


def train_learned_estimator(road_coords, all_predicted_speeds, save_path=None):
    """
    Train a gradient-boosting regressor using leave-one-road-out data.
    all_predicted_speeds: list of arrays, each (16,) for different snapshots/days.
    Returns the trained model and training metrics.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    config = load_config()
    k = config["speed_estimator"]["k_nearest"]

    X_all, y_all = [], []
    for speeds in all_predicted_speeds:
        for day_offset in range(3):
            for hold_out_road in road_coords:
                ho_idx = hold_out_road["index"]
                # Build features using all OTHER roads' speeds
                other_coords = [r for r in road_coords if r["index"] != ho_idx]
                other_speeds_map = {r["index"]: speeds[r["index"]] for r in other_coords}
                # Re-index for feature building
                reindexed_speeds = list(other_speeds_map.values())
                feats = build_features(
                    hold_out_road["lat"], hold_out_road["lng"],
                    other_coords,
                    # Use full speeds array but IDW will only use other_coords
                    speeds,
                    day_offset, k=min(k, len(other_coords))
                )
                X_all.append(feats)
                y_all.append(speeds[ho_idx])

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, min_samples_leaf=3
    )
    model.fit(X_all, y_all)

    if save_path is None:
        save_path = os.path.join(_MODEL_DIR, "speed_estimator_gbr.joblib")
    joblib.dump(model, save_path)
    print(f"[speed_estimator] Learned model saved to {save_path}")

    return model


def learned_speed(lat, lng, road_coords, predicted_speeds, day_offset,
                  model, k=5, min_speed=5.0):
    """Predict speed at (lat, lng) using the trained GBR model."""
    feats = build_features(lat, lng, road_coords, predicted_speeds, day_offset, k=k)
    pred = model.predict(feats.reshape(1, -1))[0]
    return max(pred, min_speed)


# ── Unified speed_at() interface ──────────────────────────────────────────────

class SpeedEstimator:
    """
    Unified interface for speed estimation at arbitrary points.
    Loads config and optionally the learned model on init.
    """
    def __init__(self, road_coords=None, method=None):
        self.config = load_config()
        est_cfg = self.config["speed_estimator"]

        self.method = method or est_cfg["method"]
        self.power = est_cfg["idw_power"]
        self.k = est_cfg["k_nearest"]
        self.min_speed = est_cfg["min_speed_kmh"]
        self.coverage_radius = est_cfg["coverage_radius_km"]
        self.road_coords = road_coords or load_road_coords()
        self.learned_model = None

        if self.method == "learned":
            model_path = os.path.join(_MODEL_DIR, "speed_estimator_gbr.joblib")
            if os.path.exists(model_path):
                self.learned_model = joblib.load(model_path)
            else:
                print(f"[SpeedEstimator] Warning: learned model not found at "
                      f"{model_path}, falling back to IDW")
                self.method = "idw"

    def speed_at(self, lat, lng, predicted_speeds, day_offset=0):
        """
        Estimate speed at an arbitrary (lat, lng) point.
        Returns (speed_kmh, coverage_warning).
        """
        # Check coverage: distance to nearest road
        min_dist = min(
            haversine_km(lat, lng, r["lat"], r["lng"])
            for r in self.road_coords
        )
        coverage_warning = min_dist > self.coverage_radius

        if coverage_warning:
            # Too far from any road — use mean predicted speed
            speed = float(np.mean(predicted_speeds))
            return max(speed, self.min_speed), True

        if self.method == "learned" and self.learned_model is not None:
            speed = learned_speed(
                lat, lng, self.road_coords, predicted_speeds,
                day_offset, self.learned_model, k=self.k,
                min_speed=self.min_speed
            )
        else:
            speed = idw_speed(
                lat, lng, self.road_coords, predicted_speeds,
                power=self.power, k=self.k, min_speed=self.min_speed
            )
        return speed, False


# ── Leave-one-road-out cross-validation ───────────────────────────────────────

def leave_one_out_cv(road_coords, predicted_speeds_list):
    """
    Evaluate IDW, learned, and global-mean baselines via leave-one-road-out CV.
    predicted_speeds_list: list of (16,) arrays (one per test snapshot).
    Returns dict with metrics per method.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    config = load_config()
    k = config["speed_estimator"]["k_nearest"]
    power = config["speed_estimator"]["idw_power"]
    min_speed = config["speed_estimator"]["min_speed_kmh"]

    results = {"idw": [], "learned": [], "global_mean": []}

    for speeds in predicted_speeds_list:
        global_mean = float(np.mean(speeds))

        for hold_idx in range(len(road_coords)):
            held_road = road_coords[hold_idx]
            true_speed = speeds[held_road["index"]]

            # Global mean baseline
            results["global_mean"].append((true_speed, global_mean))

            # IDW: exclude held-out road
            other_coords = [r for r in road_coords if r["index"] != held_road["index"]]
            idw_pred = idw_speed(
                held_road["lat"], held_road["lng"],
                other_coords, speeds,
                power=power, k=min(k, len(other_coords)),
                min_speed=min_speed
            )
            results["idw"].append((true_speed, idw_pred))

    # Learned: train k-fold style (train on 15, predict held-out)
    for speeds in predicted_speeds_list:
        for hold_idx in range(len(road_coords)):
            held_road = road_coords[hold_idx]
            true_speed = speeds[held_road["index"]]
            other_coords = [r for r in road_coords if r["index"] != held_road["index"]]

            # Build training data from other roads
            X_train, y_train = [], []
            for r in other_coords:
                feats = build_features(
                    r["lat"], r["lng"], other_coords, speeds,
                    day_offset=0, k=min(k, len(other_coords))
                )
                X_train.append(feats)
                y_train.append(speeds[r["index"]])

            if len(X_train) < 5:
                results["learned"].append((true_speed, float(np.mean(speeds))))
                continue

            gbr = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, min_samples_leaf=2
            )
            gbr.fit(np.array(X_train), np.array(y_train))

            feats_test = build_features(
                held_road["lat"], held_road["lng"], other_coords, speeds,
                day_offset=0, k=min(k, len(other_coords))
            )
            pred = gbr.predict(feats_test.reshape(1, -1))[0]
            pred = max(pred, min_speed)
            results["learned"].append((true_speed, pred))

    # Compute metrics
    metrics = {}
    for method_name, pairs in results.items():
        trues = np.array([p[0] for p in pairs])
        preds = np.array([p[1] for p in pairs])
        mae = float(np.abs(trues - preds).mean())
        rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
        mask = np.abs(trues) > 1.0
        mape = float(np.mean(np.abs((trues[mask] - preds[mask]) / trues[mask])) * 100) if mask.any() else 0.0
        metrics[method_name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}

    return metrics


if __name__ == "__main__":
    # Quick test: IDW estimation near a known road
    roads = load_road_coords()
    # Fake predicted speeds
    speeds = [35.0 + i * 2 for i in range(16)]
    est = SpeedEstimator(road_coords=roads, method="idw")
    s, warn = est.speed_at(12.97, 77.65, speeds, day_offset=0)
    print(f"IDW speed at (12.97, 77.65): {s:.1f} km/h, coverage_warning={warn}")

    # Far away point
    s2, warn2 = est.speed_at(13.5, 78.0, speeds, day_offset=0)
    print(f"IDW speed at (13.5, 78.0): {s2:.1f} km/h, coverage_warning={warn2}")
    assert warn2, "Should have coverage warning for far-away point"
    print("\nSpeed estimator: OK")
