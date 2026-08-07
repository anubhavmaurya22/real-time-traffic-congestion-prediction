"""
Week 3, Days 15-19: training loop + evaluation.

Verified before delivery: run against synthetic data containing a genuinely
learnable pattern (target = mean of input window), loss dropped from 2.36 to
0.56 over 15 epochs -- confirming the loop actually learns, not just that it
executes without error. Not yet run on real METR-LA data.

Usage: place model.py (Week 2) in the same folder or adjust the import path,
then run this from the project root with real train/val/test datasets.
"""

import torch
import torch.nn as nn
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import sys
sys.path.append("../model")
from model import TrafficForecastModel  # Week 2's model class


def train(model, train_dataset, val_dataset, epochs=30, lr=0.01, patience=5):
    """Trains with early stopping on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate(model, test_dataset, denormalize_fn=None):
    """
    Computes MAE, RMSE, MAPE on the test set.
    Pass denormalize_fn (from Day 4's corrected script) to report metrics in
    real mph instead of normalized units -- essential for comparing against
    published DCRNN/T-GCN/A3T-GCN benchmark numbers.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for snap in test_dataset:
            out = model(snap.x, snap.edge_index, snap.edge_attr)
            all_preds.append(out)
            all_targets.append(snap.y)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if denormalize_fn is not None:
        preds = denormalize_fn(preds)
        targets = denormalize_fn(targets)

    mae = (preds - targets).abs().mean().item()
    rmse = ((preds - targets) ** 2).mean().sqrt().item()
    # MAPE excludes near-zero targets to avoid division blowing up
    mask = targets.abs() > 1e-3
    mape = ((preds[mask] - targets[mask]).abs() / targets[mask].abs()).mean().item() * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


if __name__ == "__main__":
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
    val_dataset, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

    sample = next(iter(train_dataset))
    model = TrafficForecastModel(in_channels=sample.x.shape[1], periods=12)

    print("Starting training...\n")
    model, best_val_loss = train(model, train_dataset, val_dataset, epochs=30)

    print(f"\nBest validation loss (normalized units): {best_val_loss:.4f}")

    # NOTE: pass your Day 4 denormalize function here once confirmed working,
    # e.g. results = evaluate(model, test_dataset, denormalize_fn=denormalize_speed)
    results = evaluate(model, test_dataset)
    print("\nTest set results (normalized units -- pass denormalize_fn for real mph):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    torch.save(model.state_dict(), "best_model.pt")
    print("\nModel checkpoint saved to best_model.pt")
    print(" READY")