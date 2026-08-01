"""
Day 4 (corrected): recover real-unit denormalization stats.

IMPORTANT CORRECTION: METRLADatasetLoader already z-score normalizes the
data internally before returning it (confirmed directly from its source --
see the "Normalise as in DCRNN paper" comment in metr_la.py). This is why
Day 4's original script measured mean~0.04, std~0.965 on data it assumed
was raw -- that WAS the already-normalized data, and the round-trip check
just confirmed the normalization is internally consistent, not that the
values were raw mph.

This also means the library's normalization stats are computed globally
over train+val+test combined, before any split happens -- a mild data
leakage that's baked into the standard loader and shared by most published
METR-LA benchmarks (DCRNN, T-GCN, etc. all use this same loader). Worth a
one-line mention in your report's limitations section; it won't hurt
comparability with published numbers since they share the same setup.

The loader computes means/stds internally but does NOT expose them as
accessible attributes in the get_dataset() path -- they're discarded after
use. This script recomputes them directly from the raw node_values.npy file
the loader already downloaded, so real-mph denormalization is available for
Week 3's evaluation metrics (MAE/RMSE should be reported in real mph, not
z-score units, to be interpretable and comparable to published numbers).
"""

import numpy as np
import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

# Recompute the same means/stds the loader used internally, straight from
# the raw file it already downloaded into data/raw_data_dir (default: ./data).
raw = np.load("node_values.npy").transpose((1, 2, 0)).astype(np.float32)
means = raw.mean(axis=(0, 2))   # per-sensor mean, shape (207,)
stds = raw.std(axis=(0, 2))     # per-sensor std,  shape (207,)

print(f"recovered means shape: {means.shape}, sample: {means[:5].round(2)}")
print(f"recovered stds shape:  {stds.shape}, sample: {stds[:5].round(2)}")
print(f"overall real-speed range in raw data: {raw.min():.1f} to {raw.max():.1f}")

means_t = torch.tensor(means, dtype=torch.float32)
stds_t = torch.tensor(stds, dtype=torch.float32)


def denormalize_speed(x_normalized, means=means_t, stds=stds_t):
    """x_normalized: shape (nodes, timesteps) -- means/stds broadcast over
    the node axis (dim 0)."""
    return x_normalized * stds.unsqueeze(-1) + means.unsqueeze(-1)


# Verify: take a normalized window from the dataset, denormalize it, and
# confirm the result falls in a realistic mph range (roughly 0-70).
train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
sample = next(iter(train_dataset))
normalized_speed = sample.x[:, 0, :]   # (nodes, timesteps)
recovered_speed = denormalize_speed(normalized_speed)
print(f"\nsample denormalized speed range: {recovered_speed.min():.1f} to {recovered_speed.max():.1f} mph")
print("(sanity check: LA highway speeds should fall roughly in 0-70 mph)")

print("\nDay 4 (corrected): READY")
print("Use denormalize_speed() in Week 3's evaluation to report MAE/RMSE in real mph.")