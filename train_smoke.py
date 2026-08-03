import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
import torch
import torch.nn as nn
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from model import TrafficForecastModel

_data_dir = os.path.join(os.path.dirname(__file__), "model", "data")
loader = METRLADatasetLoader(raw_data_dir=_data_dir)
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
train_dataset, remaining = temporal_signal_split(dataset, train_ratio=0.7)
val_dataset, test_dataset = temporal_signal_split(remaining, train_ratio=1/3)

train_slice = list(train_dataset)[:300]
val_slice = list(val_dataset)[:100]

sample = train_slice[0]
model = TrafficForecastModel(in_channels=sample.x.shape[1], periods=12)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"Training on {len(train_slice)} windows, validating on {len(val_slice)}\n")
start = time.time()

for epoch in range(2):
    model.train()
    train_loss = 0.0
    for i, snap in enumerate(train_slice):
        optimizer.zero_grad()
        out = model(snap.x, snap.edge_index, snap.edge_attr)
        loss = nn.functional.l1_loss(out, snap.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i + 1) % 50 == 0:
            print(f"  epoch {epoch+1}, window {i+1}/{len(train_slice)}, elapsed {time.time()-start:.1f}s")

    train_loss /= len(train_slice)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for snap in val_slice:
            out = model(snap.x, snap.edge_index, snap.edge_attr)
            val_loss += nn.functional.l1_loss(out, snap.y).item()
    val_loss /= len(val_slice)
    print(f"epoch {epoch+1} DONE  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  total_elapsed {time.time()-start:.1f}s\n")

per_window = (time.time()-start) / (2 * len(train_slice))
print(f"~{per_window*1000:.1f} ms/window -> full epoch (~24000 windows) would take ~{per_window*24000/60:.1f} min")
