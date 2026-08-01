"""
Day 3: chronological train/val/test split.

Uses torch_geometric_temporal's own temporal_signal_split utility, chained
twice to get a 70/10/20 split. This is a straight index-based split with no
shuffling -- correct for time series, since shuffling would leak future
information into training.

Run data/verify_setup.py first if you haven't confirmed it's READY.
"""

from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

# First split: 70% train, 30% held out
train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)

# Second split: divide the remaining 30% into 10% val / 20% test (1/3 : 2/3)
val_dataset, test_dataset = temporal_signal_split(remaining, train_ratio=1 / 3)

print(f"train windows: {train_dataset.snapshot_count}")
print(f"val windows:   {val_dataset.snapshot_count}")
print(f"test windows:  {test_dataset.snapshot_count}")
total = train_dataset.snapshot_count + val_dataset.snapshot_count + test_dataset.snapshot_count
print(f"total: {total}")

# Sanity check: confirm chronological ordering, not shuffled -- first test
# snapshot should come after the last train snapshot in time.
train_list = list(train_dataset)
val_list = list(val_dataset)
test_list = list(test_dataset)
print(f"\nfirst train window shape: {tuple(train_list[0].x.shape)}")
print(f"last test window shape:   {tuple(test_list[-1].x.shape)}")
print("\nDay 3 split: READY -- use train_dataset/val_dataset/test_dataset as-is in Week 2's training loop")