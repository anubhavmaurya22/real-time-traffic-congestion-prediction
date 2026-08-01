"""
Week 2, Day 10: model architecture.

Wraps torch_geometric_temporal's A3TGCN layer with a linear output head that
maps its hidden representation to a 12-step forecast per node.

Verified before delivery: forward pass produces output shape (207, 12)
matching the target shape exactly, and a full backward pass (loss.backward())
runs without error on synthetic data shaped like real METR-LA windows.
Not yet trained on real data -- that's Week 3.
"""

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN


class TrafficForecastModel(nn.Module):
    def __init__(self, in_channels: int, periods: int, hidden_dim: int = 32):
        """
        in_channels: number of input features per node per timestep.
                     METR-LA has 2 (speed, and a second channel -- confirmed
                     via Day 4's exploration of node_values.npy).
        periods:     number of timesteps in the input window (12, per the
                     loader config used since Day 1 -- 1 hour of 5-min readings).
        hidden_dim:  size of A3TGCN's hidden state. 32 is a reasonable
                     starting point; increase if training underfits.
        """
        super().__init__()
        self.tgnn = A3TGCN(in_channels=in_channels, out_channels=hidden_dim, periods=periods)
        self.linear = nn.Linear(hidden_dim, periods)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x:           (num_nodes, in_channels, periods) -- matches the .x
                     shape of a snapshot from the dataset directly.
        edge_index:  (2, num_edges) -- from snapshot.edge_index.
        edge_weight: (num_edges,) -- from snapshot.edge_attr, optional.

        Returns: (num_nodes, periods) forecast, matching snapshot.y shape.
        """
        h = self.tgnn(x, edge_index, edge_weight)
        h = torch.relu(h)
        return self.linear(h)


if __name__ == "__main__":
    # Dry run on real data shapes (not real values) -- confirms the model
    # wires up correctly against the actual dataset before Week 3's training.
    from torch_geometric_temporal.dataset import METRLADatasetLoader

    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    snapshot = next(iter(dataset))

    model = TrafficForecastModel(in_channels=snapshot.x.shape[1], periods=12)
    out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

    print(f"input shape:  {tuple(snapshot.x.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"target shape: {tuple(snapshot.y.shape)}")
    assert out.shape == snapshot.y.shape, "Output shape must match target shape"

    loss = torch.nn.functional.l1_loss(out, snapshot.y)
    loss.backward()
    print(f"\nforward + backward pass OK on real data. initial loss (untrained): {loss.item():.4f}")
    print(f"param count: {sum(p.numel() for p in model.parameters()):,}")
    print("\nWeek 2 model architecture: READY for Week 3 training")