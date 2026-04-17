from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import NNConv, global_mean_pool


class GraphMP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int = 0,
        edge_dim: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        hidden_channels = out_channels
        readout_input_dim = hidden_channels + global_dim
        edge_network1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels * hidden_channels),
        )
        edge_network2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * hidden_channels),
        )
        self.conv1 = NNConv(in_channels, hidden_channels, nn=edge_network1, aggr="mean")
        self.conv2 = NNConv(hidden_channels, hidden_channels, nn=edge_network2, aggr="mean")
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = float(dropout)
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, global_features=None):
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), 1))

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        if global_features is not None:
            x = torch.cat([x, global_features], dim=1)
        return self.readout(x)
