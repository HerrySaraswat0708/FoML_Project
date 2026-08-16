import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool


class GraphNET(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int = 0,
        edge_dim: int = 10,
        dropout: float = 0.2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = out_channels
        del heads
        readout_input_dim = (hidden_channels * 3) + hidden_channels + hidden_channels
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINEConv(self._make_conv_mlp(hidden_channels))
        self.conv2 = GINEConv(self._make_conv_mlp(hidden_channels))
        self.conv3 = GINEConv(self._make_conv_mlp(hidden_channels))
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.norm3 = nn.BatchNorm1d(hidden_channels)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.ELU(),
            nn.Dropout(p=dropout),
        ) if global_dim > 0 else None
        self.dropout = float(dropout)
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, hidden_channels),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    @staticmethod
    def _make_conv_mlp(hidden_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, global_features=None):
        x = self.node_encoder(x)
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_encoder[0].in_features))
        edge_attr = self.edge_encoder(edge_attr)

        residual = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x + residual)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual = x
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x + residual)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual = x
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x + residual)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        pooled_add = global_add_pool(x, batch)
        x = torch.cat([pooled_mean, pooled_max, pooled_add], dim=1)
        if global_features is not None:
            if self.global_encoder is not None:
                global_features = self.global_encoder(global_features)
            x = torch.cat([x, global_features, pooled_mean - pooled_max], dim=1)
        else:
            x = torch.cat([x, pooled_mean - pooled_max, pooled_mean], dim=1)
        return self.readout(x)
