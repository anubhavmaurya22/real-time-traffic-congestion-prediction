"""
Bangalore dataset training + evaluation.

Verified end-to-end with synthetic data matching the real dataset's
structure (16 nodes, 952 timesteps, 7-day input -> 3-day forecast) before
delivery -- training loop runs cleanly, loss updates each epoch, early
stopping and checkpoint saving work correctly.

Uses the corrected model.py (separate in_periods/out_periods) -- this
dataset's 7-in/3-out window would break the original METR-LA-only version.

Denormalizes using the means/stds returned directly by load_bangalore_dataset()
-- more reliable than hardcoding values by hand (as the METR-LA pipeline
did), since these are computed fresh from whatever data is actually loaded.
"""

import torch
import torch.nn as nn
from bangalore_data import load_bangalore_dataset
from torch_geometric_temporal.signal import temporal_signal_split

import sys
sys.path.append("../model")
from model import TrafficForecastModel


def train(model, train_dataset, val_dataset, epochs=50, lr=0.01, patience=8):
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
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate(model, test_dataset, speed_mean, speed_std):
    """Denormalizes using the real per-dataset mean/std (feature index 0 =
    Average Speed) returned by load_bangalore_dataset() -- reports MAE/RMSE/
    MAPE in real km/h, not normalized units."""
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
    dataset, means, stds, road_order = load_bangalore_dataset(num_timesteps_in=7, num_timesteps_out=3)
    train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
    val_dataset, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

    speed_mean = float(means[0, 0, 0])  # feature index 0 = Average Speed
    speed_std = float(stds[0, 0, 0])

    sample = next(iter(train_dataset))
    model = TrafficForecastModel(in_channels=sample.x.shape[1], in_periods=7, out_periods=3)

    print(f"\nTraining on {train_dataset.snapshot_count} windows "
          f"({len(road_order)} roads, 7-day input -> 3-day forecast)\n")
    model, best_val_loss = train(model, train_dataset, val_dataset, epochs=50, patience=8)

    print(f"\nBest validation loss (normalized units): {best_val_loss:.4f}")

    results = evaluate(model, test_dataset, speed_mean, speed_std)
    print("\nTest set results (real km/h):")
    for k, v in results.items():
        unit = "km/h" if k != "MAPE" else "%"
        print(f"  {k}: {v:.2f} {unit}")

    # NAIVE BASELINE COMPARISON -- important context given this dataset's small
    # size (660 training windows, 16 nodes). Predicting the mean every time
    # gives a theoretical MAE of sqrt(2/pi) =~ 0.798 in normalized units --
    # if your model's loss is close to this, it's barely beating a baseline
    # that ignores the input entirely. Report this comparison honestly.
    model.eval()
    naive_mean_preds, naive_last_preds, all_targets = [], [], []
    with torch.no_grad():
        for snap in test_dataset:
            # naive baseline 1: predict the (normalized) mean, i.e. 0, for every horizon
            naive_mean_preds.append(torch.zeros_like(snap.y))
            # naive baseline 2: predict the last observed value repeated across the horizon
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
        print("  This is a real, reportable finding given the dataset's small size")
        print("  (660 training windows, 16 nodes) -- worth discussing honestly in")
        print("  your report's limitations/future work section, e.g. more data,")
        print("  more roads, or a simpler model may be needed for this dataset scale.")


    torch.save(model.state_dict(), "bangalore_best_model.pt")
    print("\nModel checkpoint saved to bangalore_best_model.pt")
    print("Bangalore training + evaluation: READY")