"""
Day 1 setup check.

Run this after installing requirements.txt to confirm your environment is
ready. It intentionally does NOT explore the data yet (that's Day 2) —
it just proves the pipeline is wired correctly end to end:
    imports -> data download -> graph + tensors in hand.

Expected output ends with "Day 1 environment: READY".
"""

import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN

print(f"torch {torch.__version__}")

# Downloads METR-LA on first run (~few minutes), caches locally after that.
loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

# StaticGraphTemporalSignal is iterable; grab one snapshot to sanity-check shapes.
snapshot = next(iter(dataset))
print(f"nodes (sensors): {snapshot.x.shape[0]}")
print(f"input window shape:  {tuple(snapshot.x.shape)}   # (nodes, features, 12 timesteps in)")
print(f"target window shape: {tuple(snapshot.y.shape)}   # (nodes, 12 timesteps out)")
print(f"edges in sensor graph: {snapshot.edge_index.shape[1]}")

# Confirm the model class is usable (not training yet — that's Week 3).
model = A3TGCN(in_channels=snapshot.x.shape[1], out_channels=32, periods=12)
print(f"A3TGCN parameter count: {sum(p.numel() for p in model.parameters()):,}")

print("\nDay 1 environment: READY")