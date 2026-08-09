import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

import torch
import numpy as np
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from model import TrafficForecastModel

_data_dir = os.path.join(os.path.dirname(__file__), "model", "data")
loader = METRLADatasetLoader(raw_data_dir=_data_dir)
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
val_dataset, test_dataset = temporal_signal_split(remaining, train_ratio=1/3)

sample = next(iter(train_dataset))
model = TrafficForecastModel(in_channels=sample.x.shape[1], in_periods=12, out_periods=12)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model", "best_model.pt")))
model.eval()

speed_mean, speed_std = 53.6, 20.16

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

print(f"MAE:  {mae:.2f} mph")
print(f"RMSE: {rmse:.2f} mph")
print(f"MAPE: {mape:.2f}%")
