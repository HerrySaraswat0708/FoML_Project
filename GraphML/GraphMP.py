import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool

class GraphMP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.msg_lin = Linear(2 * in_channels, out_channels)
        self.update_lin = Linear(out_channels, out_channels)
        self.linf = Linear(out_channels, 1)

    def forward(self, x, edge_index, batch):
        out = self.propagate(edge_index, x=x)
        out = global_mean_pool(out, batch)
        out = self.update_lin(out)
        out = self.linf(out)
        return out

    def message(self, x_i, x_j):
        # Combine source and target
        msg = torch.cat([x_i, x_j], dim=1)
        return self.msg_lin(msg)