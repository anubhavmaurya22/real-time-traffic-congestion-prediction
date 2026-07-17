"""
Day 2: explore the raw METR-LA data.
Run data/verify_setup.py first if you haven't confirmed it's READY.
"""

import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader

loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

snapshots = list(dataset)
print(f"total time windows: {len(snapshots)}")

first = snapshots[0]
print(f"raw x shape (one window): {tuple(first.x.shape)}")
print(f"raw y shape (one window): {tuple(first.y.shape)}")
print(f"sensors (nodes): {first.x.shape[0]}")

# Speed is feature index 0. Sample the first 500 windows to check range + missing values
# without loading everything into memory at once.
sample = torch.stack([s.x[:, 0, :] for s in snapshots[:500]])
print(f"speed range (sample): {sample.min():.2f} to {sample.max():.2f}")
print(f"zero-reading fraction (likely missing/offline sensors): {(sample == 0).float().mean():.3%}")