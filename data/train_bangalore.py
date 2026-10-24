"""
Bangalore dataset training — reads config.yaml, seeds for reproducibility,
logs per-epoch loss to training_log.csv.

Usage:
    cd data
    python train_bangalore.py
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
import yaml

from bangalore_data import load_bangalore_dataset
from torch_geometric_temporal.signal import temporal_signal_split

sys.path.append("../model")
from model import TrafficForecastModel

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_BASE_DIR, "model")
_DATA_DIR = os.path.join(_BASE_DIR, "data")


def load_config():
    """Load model/config.yaml."""
    config_path = os.path.join(_MODEL_DIR, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(model, train_dataset, val_dataset, epochs=50, lr=0.01, patience=8,
          log_path=None):
    """
    Train with early stopping. Writes per-epoch loss to log_path (CSV).
    Returns (model, best_val_loss).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    # Open CSV log writer
    log_file = None
    log_writer = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "train_loss", "val_loss"])

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for snap in train_dataset:
                optimizer.zero_grad()
                out = model(snap.x, snap.edge_index, snap.edge_attr)
                loss = nn.functional.l1_loss(out, snap.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= train_dataset.snapshot_count

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for snap in val_dataset:
                    out = model(snap.x, snap.edge_index, snap.edge_attr)
                    val_loss += nn.functional.l1_loss(out, snap.y).item()
            val_loss /= val_dataset.snapshot_count

            print(f"epoch {epoch+1:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

            if log_writer:
                log_writer.writerow([epoch + 1, round(train_loss, 6), round(val_loss, 6)])
                log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    finally:
        if log_file:
            log_file.close()

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate(model, test_dataset, speed_mean, speed_std):
    """Evaluate on test set — reports MAE/RMSE/MAPE in real km/h."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for snap in test_dataset:
            out = model(snap.x, snap.edge_index, snap.edge_attr)
            all_preds.append(out)
            all_targets.append(snap.y)

    preds = torch.cat(all_preds, dim=0) * speed_std + speed_mean
    targets = torch.cat(all_targets, dim=0) * speed_std + speed_mean

    mae = (preds - targets).abs().mean().item()
    rmse = ((preds - targets) ** 2).mean().sqrt().item()
    mask = targets.abs() > 1.0
    mape = ((preds[mask] - targets[mask]).abs() / targets[mask].abs()).mean().item() * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


if __name__ == "__main__":
    config = load_config()

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[train] seed={seed}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset, means, stds, road_order, all_dates = load_bangalore_dataset(
        num_timesteps_in=config["data"]["num_timesteps_in"],
        num_timesteps_out=config["data"]["num_timesteps_out"],
    )
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"].get("val_ratio_of_remaining", 1 / 3)
    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=train_ratio)
    val_dataset, test_dataset = temporal_signal_split(remaining, train_ratio=val_ratio)

    speed_mean = float(means[0, 0, 0])
    speed_std = float(stds[0, 0, 0])

    # ── Model ────────────────────────────────────────────────────────────────
    sample = next(iter(train_dataset))
    model = TrafficForecastModel(
        in_channels=sample.x.shape[1],
        in_periods=config["data"]["num_timesteps_in"],
        out_periods=config["data"]["num_timesteps_out"],
        hidden_dim=config["model"]["hidden_dim"],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    log_path = os.path.join(_MODEL_DIR, "training_log.csv")
    print(f"\nTraining on {train_dataset.snapshot_count} windows "
          f"({len(road_order)} roads, {config['data']['num_timesteps_in']}-day input "
          f"-> {config['data']['num_timesteps_out']}-day forecast)\n")

    model, best_val_loss = train(
        model, train_dataset, val_dataset,
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"],
        patience=config["training"]["patience"],
        log_path=log_path,
    )
    print(f"\nBest validation loss (normalized): {best_val_loss:.4f}")
    print(f"Training log saved to {log_path}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results = evaluate(model, test_dataset, speed_mean, speed_std)
    print("\nTest set results (real km/h):")
    for k, v in results.items():
        unit = "km/h" if k != "MAPE" else "%"
        print(f"  {k}: {v:.2f} {unit}")

    # ── Baseline comparison ──────────────────────────────────────────────────
    model.eval()
    naive_mean_preds, naive_last_preds, all_targets = [], [], []
    with torch.no_grad():
        for snap in test_dataset:
            naive_mean_preds.append(torch.zeros_like(snap.y))
            last_val = snap.x[:, 0, -1].unsqueeze(-1).repeat(1, snap.y.shape[1])
            naive_last_preds.append(last_val)
            all_targets.append(snap.y)

    naive_mean_preds = torch.cat(naive_mean_preds, dim=0) * speed_std + speed_mean
    naive_last_preds = torch.cat(naive_last_preds, dim=0) * speed_std + speed_mean
    targets_real = torch.cat(all_targets, dim=0) * speed_std + speed_mean

    naive_mean_mae = (naive_mean_preds - targets_real).abs().mean().item()
    naive_last_mae = (naive_last_preds - targets_real).abs().mean().item()

    print("\nBaseline comparison (real km/h MAE):")
    print(f"  Your model:              {results['MAE']:.2f}")
    print(f"  Naive (predict mean):    {naive_mean_mae:.2f}")
    print(f"  Naive (predict last obs): {naive_last_mae:.2f}")
    if results['MAE'] >= naive_mean_mae * 0.95:
        print("\n  NOTE: model's MAE is close to or worse than the naive mean baseline.")
        print("  This is a real finding given the dataset's small size — discuss")
        print("  honestly in your report's limitations section.")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(_DATA_DIR, "bangalore_best_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel checkpoint saved to {ckpt_path}")

    print("Bangalore training: DONE")