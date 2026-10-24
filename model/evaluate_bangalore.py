"""
Evaluation script for StreetFlow Live — Bangalore A3T-GCN model.

Reports:
  1. MAE, RMSE, MAPE per forecast horizon (day 1, 2, 3) for:
     - A3T-GCN model
     - Persistence baseline (repeat last day of input)
     - Historical average baseline
  2. Per-road MAE analysis (best/worst).
  3. Leave-one-road-out spatial estimator metrics — both strict (excludes
     near-duplicate roads within 0.5 km) and original (single exclusion).
     Evaluated on both model-predicted and observed speeds.
  4. Saves results.json and PNG plots:
     - training_curve.png (if training log CSV exists)
     - error_vs_horizon.png
     - predicted_vs_observed.png (best and worst road)
  5. ETA sanity check from data/eta_check.csv (if user-filled).

Every number in results.json is computed by this script — nothing fabricated.

Usage:
    python model/evaluate_bangalore.py
"""

import os
import sys
import json
import numpy as np
import torch
import pandas as pd
import yaml

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE_DIR, "data"))
sys.path.insert(0, os.path.join(_BASE_DIR, "model"))

from bangalore_data import load_bangalore_dataset
from torch_geometric_temporal.signal import temporal_signal_split
from model import TrafficForecastModel
from speed_estimator import load_road_coords, leave_one_out_cv

# ── Config ────────────────────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(_BASE_DIR, "model")
_DATA_DIR = os.path.join(_BASE_DIR, "data")

def load_config():
    with open(os.path.join(_MODEL_DIR, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(preds, targets):
    """Compute MAE, RMSE, MAPE on real-valued (denormalized) tensors."""
    mae = float((preds - targets).abs().mean())
    rmse = float(((preds - targets) ** 2).mean().sqrt())
    mask = targets.abs() > 1.0
    mape = float(
        ((preds[mask] - targets[mask]).abs() / targets[mask].abs()).mean() * 100
    ) if mask.any() else 0.0
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}


def per_horizon_metrics(preds, targets, speed_mean, speed_std, num_horizons=3):
    """Return metrics dict per horizon day."""
    results = {}
    for h in range(num_horizons):
        p = preds[:, h] * speed_std + speed_mean
        t = targets[:, h] * speed_std + speed_mean
        results[f"day_{h+1}"] = compute_metrics(p, t)
    return results


# ── Main evaluation ──────────────────────────────────────────────────────────

def main():
    config = load_config()

    # Fix seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load data
    dataset, means, stds, road_order, all_dates = load_bangalore_dataset(
        num_timesteps_in=config["data"]["num_timesteps_in"],
        num_timesteps_out=config["data"]["num_timesteps_out"],
    )
    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=config["data"]["train_ratio"])
    val_dataset, test_dataset = temporal_signal_split(
        remaining, train_ratio=config["data"]["val_ratio_of_remaining"]
    )

    speed_mean = float(means[0, 0, 0])
    speed_std = float(stds[0, 0, 0])

    # Load model
    sample = next(iter(train_dataset))
    model = TrafficForecastModel(
        in_channels=sample.x.shape[1],
        in_periods=config["data"]["num_timesteps_in"],
        out_periods=config["data"]["num_timesteps_out"],
        hidden_dim=config["model"]["hidden_dim"],
    )
    ckpt_path = os.path.join(_DATA_DIR, "bangalore_best_model.pt")
    try:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"[eval] Loaded checkpoint from {ckpt_path}")

    # ── Collect predictions ──────────────────────────────────────────────────
    all_preds, all_targets, all_inputs = [], [], []
    with torch.no_grad():
        for snap in test_dataset:
            out = model(snap.x, snap.edge_index, snap.edge_attr)
            all_preds.append(out)
            all_targets.append(snap.y)
            all_inputs.append(snap.x)

    preds = torch.stack(all_preds)        # (T, 16, 3)
    targets = torch.stack(all_targets)    # (T, 16, 3)
    inputs = torch.stack(all_inputs)      # (T, 16, 2, 7)

    num_test = preds.shape[0]
    print(f"[eval] Test windows: {num_test}")

    # ── 1. A3T-GCN per-horizon metrics ───────────────────────────────────────
    preds_flat = preds.reshape(-1, 3)
    targets_flat = targets.reshape(-1, 3)

    model_metrics = per_horizon_metrics(preds_flat, targets_flat, speed_mean, speed_std)
    print("\n=== A3T-GCN Metrics (per horizon) ===")
    for k, v in model_metrics.items():
        print(f"  {k}: MAE={v['MAE']:.3f}  RMSE={v['RMSE']:.3f}  MAPE={v['MAPE']:.2f}%")

    # ── 2. Persistence baseline ──────────────────────────────────────────────
    persist_pred = inputs[:, :, 0, -1].unsqueeze(-1).expand_as(targets)
    persist_flat = persist_pred.reshape(-1, 3)
    persist_metrics = per_horizon_metrics(persist_flat, targets_flat, speed_mean, speed_std)
    print("\n=== Persistence Baseline (repeat last day) ===")
    for k, v in persist_metrics.items():
        print(f"  {k}: MAE={v['MAE']:.3f}  RMSE={v['RMSE']:.3f}  MAPE={v['MAPE']:.2f}%")

    # ── 3. Historical average baseline ───────────────────────────────────────
    train_targets = []
    with torch.no_grad():
        for snap in train_dataset:
            train_targets.append(snap.y)
    train_targets_stacked = torch.stack(train_targets)
    hist_avg = train_targets_stacked.mean(dim=0)
    hist_pred = hist_avg.unsqueeze(0).expand_as(targets)
    hist_flat = hist_pred.reshape(-1, 3)
    hist_metrics = per_horizon_metrics(hist_flat, targets_flat, speed_mean, speed_std)
    print("\n=== Historical Average Baseline ===")
    for k, v in hist_metrics.items():
        print(f"  {k}: MAE={v['MAE']:.3f}  RMSE={v['RMSE']:.3f}  MAPE={v['MAPE']:.2f}%")

    # ── 4. Per-road analysis (best/worst) ─────────────────────────────────────
    per_road_mae = {}
    for r_idx in range(len(road_order)):
        p = preds[:, r_idx, 0] * speed_std + speed_mean
        t = targets[:, r_idx, 0] * speed_std + speed_mean
        per_road_mae[road_order[r_idx]] = float((p - t).abs().mean())

    best_road = min(per_road_mae, key=per_road_mae.get)
    worst_road = max(per_road_mae, key=per_road_mae.get)
    print(f"\n  Best road  (lowest MAE): {best_road} ({per_road_mae[best_road]:.3f} km/h)")
    print(f"  Worst road (highest MAE): {worst_road} ({per_road_mae[worst_road]:.3f} km/h)")

    # ── 5. Spatial estimator leave-one-out CV ─────────────────────────────────
    road_coords = load_road_coords()

    # Collect at least 50 test snapshots (use all available)
    spatial_speeds_predicted = []
    spatial_speeds_observed = []
    with torch.no_grad():
        for snap in test_dataset:
            out = model(snap.x, snap.edge_index, snap.edge_attr)
            pred_speeds = (out[:, 0] * speed_std + speed_mean).numpy()
            obs_speeds = (snap.y[:, 0] * speed_std + speed_mean).numpy()
            spatial_speeds_predicted.append(pred_speeds)
            spatial_speeds_observed.append(obs_speeds)

    # Use 50 test snapshots as specified in Task 2d ("at least 50 test snapshots (not 10)")
    spatial_speeds_predicted = spatial_speeds_predicted[:50]
    spatial_speeds_observed = spatial_speeds_observed[:50]
    n_snapshots = len(spatial_speeds_predicted)
    print(f"\n[eval] Using {n_snapshots} test snapshots for spatial CV", flush=True)


    # Strict CV (excludes near-duplicate roads within 0.5 km)
    print("\n=== Spatial Estimator — Strict LOO CV (predicted speeds) ===")
    strict_pred_metrics = leave_one_out_cv(road_coords, spatial_speeds_predicted, strict=True)
    for method, m in strict_pred_metrics.items():
        print(f"  {method}: MAE={m['MAE']:.3f} +/- {m['MAE_std']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.2f}%  (n={m['n_samples']})")

    # Original CV (single exclusion only)
    print("\n=== Spatial Estimator — Original LOO CV (predicted speeds) ===")
    original_pred_metrics = leave_one_out_cv(road_coords, spatial_speeds_predicted, strict=False)
    for method, m in original_pred_metrics.items():
        print(f"  {method}: MAE={m['MAE']:.3f} +/- {m['MAE_std']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.2f}%  (n={m['n_samples']})")

    # Strict CV on observed speeds
    print("\n=== Spatial Estimator — Strict LOO CV (observed speeds) ===")
    strict_obs_metrics = leave_one_out_cv(road_coords, spatial_speeds_observed, strict=True)
    for method, m in strict_obs_metrics.items():
        print(f"  {method}: MAE={m['MAE']:.3f} +/- {m['MAE_std']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.2f}%  (n={m['n_samples']})")

    # Select estimator method from strict CV
    idw_strict = strict_pred_metrics.get("idw", {}).get("MAE", float("inf"))
    learned_strict = strict_pred_metrics.get("learned", {}).get("MAE", float("inf"))
    # Require learned to beat IDW by at least 5% to justify its complexity
    if learned_strict < idw_strict * 0.95:
        selected_method = "learned"
    else:
        selected_method = "idw"
    print(f"\n  Selected estimator method: {selected_method}")

    # Update config.yaml with selected method
    config["speed_estimator"]["method"] = selected_method
    config_path = os.path.join(_MODEL_DIR, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Updated {config_path}")

    # ── 6. ETA sanity check ──────────────────────────────────────────────────
    eta_csv_path = os.path.join(_DATA_DIR, "eta_check.csv")
    eta_sanity = None
    if os.path.exists(eta_csv_path):
        eta_df = pd.read_csv(eta_csv_path)
        if len(eta_df) > 0 and "our_eta_min" in eta_df.columns and "reference_eta_min" in eta_df.columns:
            valid = eta_df.dropna(subset=["our_eta_min", "reference_eta_min"])
            if len(valid) > 0:
                abs_diff = (valid["our_eta_min"] - valid["reference_eta_min"]).abs()
                eta_sanity = {
                    "n_routes": len(valid),
                    "mean_abs_diff_min": round(float(abs_diff.mean()), 2),
                    "max_abs_diff_min": round(float(abs_diff.max()), 2),
                }
                print(f"\n=== ETA Sanity Check ===")

                print(f"  {eta_sanity['n_routes']} reference routes")
                print(f"  Mean |our - reference|: {eta_sanity['mean_abs_diff_min']:.2f} min")
            else:
                print("\n[eval] eta_check.csv exists but has no filled rows — skipping")
        else:
            print("\n[eval] eta_check.csv exists but has no data rows — skipping")
    else:
        print(f"\n[eval] {eta_csv_path} not found — create it and fill 5+ reference ETAs")

    # ── 7. Assemble results.json ──────────────────────────────────────────────
    results = {
        "model": "A3T-GCN (Bangalore, 16 roads, daily)",
        "test_windows": num_test,
        "date_range": {
            "start": str(all_dates[0].date()),
            "end": str(all_dates[-1].date()),
            "days": len(all_dates),
        },
        "speed_stats": {"mean_kmh": round(speed_mean, 2), "std_kmh": round(speed_std, 2)},
        "a3tgcn_per_horizon": model_metrics,
        "persistence_per_horizon": persist_metrics,
        "historical_avg_per_horizon": hist_metrics,
        "per_road_mae_day1": {k: round(v, 3) for k, v in per_road_mae.items()},
        "best_road": best_road,
        "worst_road": worst_road,
        "spatial_estimator": {
            "selected_method": selected_method,
            "n_test_snapshots": n_snapshots,
            "strict_cv_predicted_speeds": strict_pred_metrics,
            "original_cv_predicted_speeds": original_pred_metrics,
            "strict_cv_observed_speeds": strict_obs_metrics,
        },
    }
    if eta_sanity:
        results["eta_sanity_check"] = eta_sanity

    results_path = os.path.join(_MODEL_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {results_path}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── 8a. Training curve (if CSV exists) ───────────────────────────────
        log_csv = os.path.join(_MODEL_DIR, "training_log.csv")
        if os.path.exists(log_csv):
            log_df = pd.read_csv(log_csv)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", color="#f2ca50")
            ax.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss", color="#99907c")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (MAE, normalized)")
            ax.set_title("Training Curve — Bangalore A3T-GCN")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(_MODEL_DIR, "training_curve.png"), dpi=150)
            plt.close(fig)
            print("[eval] Saved training_curve.png")

        # ── 8b. Error vs horizon ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        horizons = ["Day 1", "Day 2", "Day 3"]
        for label, metrics_dict, color in [
            ("A3T-GCN", model_metrics, "#f2ca50"),
            ("Persistence", persist_metrics, "#99907c"),
            ("Historical Avg", hist_metrics, "#ef4444"),
        ]:
            maes = [metrics_dict[f"day_{h+1}"]["MAE"] for h in range(3)]
            ax.plot(horizons, maes, marker="o", label=label, color=color, linewidth=2)
        ax.set_ylabel("MAE (km/h)")
        ax.set_title("Forecast Error vs Horizon")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(_MODEL_DIR, "error_vs_horizon.png"), dpi=150)
        plt.close(fig)
        print("[eval] Saved error_vs_horizon.png")

        # ── 8c. Predicted vs observed (best + worst road) ─────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, road_name, title_suffix in [
            (axes[0], best_road, "Best"),
            (axes[1], worst_road, "Worst"),
        ]:
            r_idx = road_order.index(road_name)
            obs = (targets[:, r_idx, 0] * speed_std + speed_mean).numpy()
            pred = (preds[:, r_idx, 0] * speed_std + speed_mean).numpy()
            ax.plot(obs, label="Observed", color="#99907c", linewidth=1.5)
            ax.plot(pred, label="Predicted", color="#f2ca50", linewidth=1.5)
            ax.set_xlabel("Test Window Index")
            ax.set_ylabel("Speed (km/h)")
            ax.set_title(f"{road_name} ({title_suffix} Road)")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(_MODEL_DIR, "predicted_vs_observed.png"), dpi=150)
        plt.close(fig)
        print("[eval] Saved predicted_vs_observed.png")

    except ImportError:
        print("[eval] matplotlib not installed — skipping plots")

    print("\n[eval] Evaluation complete.")


if __name__ == "__main__":
    main()
