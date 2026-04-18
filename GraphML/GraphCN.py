<<<<<<< HEAD
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int = 0,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        hidden_channels = out_channels
        readout_input_dim = hidden_channels + global_dim
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = float(dropout)
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch, global_features=None):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        if global_features is not None:
            x = torch.cat([x, global_features], dim=1)
        return self.readout(x)
=======