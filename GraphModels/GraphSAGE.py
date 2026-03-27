import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool

class GraphSAGE(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # mean aggregation
        self.lin = Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, batch):
        # Step 1: aggregate neighbor messages
        out = self.propagate(edge_index, x=x)

        # Step 2: concatenate self node features
        out = torch.cat([x, out], dim=1)
        out = global_mean_pool(out,batch)
        # Step 3: linear transformation
        out = self.lin(out)

        return out

    def message(self, x_j):
        # Just pass neighbor features
        return x_j