

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN


class TrafficForecastModel(nn.Module):
    def __init__(self, in_channels: int, in_periods: int, out_periods: int, hidden_dim: int = 32):
        """
        in_channels: number of input features per node per timestep.
        in_periods:  number of timesteps in the input window (e.g. 12 for
                     METR-LA's 1-hour window, 7 for Bangalore's 7-day window).
        out_periods: number of timesteps to forecast (e.g. 12 for METR-LA,
                     3 for Bangalore's 3-day forecast). Independent of
                     in_periods -- this is the fix.
        """
        super().__init__()
        self.tgnn = A3TGCN(in_channels=in_channels, out_channels=hidden_dim, periods=in_periods)
        self.linear = nn.Linear(hidden_dim, out_periods)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.tgnn(x, edge_index, edge_weight)
        h = torch.relu(h)
        return self.linear(h)


if __name__ == "__main__":
    x = torch.rand(16, 2, 7)
    edge_index = torch.randint(0, 16, (2, 100))
    edge_weight = torch.rand(100)

    model = TrafficForecastModel(in_channels=2, in_periods=7, out_periods=3)
    out = model(x, edge_index, edge_weight)
    print(f"output shape: {tuple(out.shape)} (expect (16, 3))")
    assert out.shape == (16, 3)

    loss = torch.nn.functional.l1_loss(out, torch.rand(16, 3))
    loss.backward()
    print("forward + backward pass OK")
    