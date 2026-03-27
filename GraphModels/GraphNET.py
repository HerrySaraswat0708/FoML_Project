import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool

class GraphNET(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')

        self.lin = Linear(in_channels, out_channels, bias=False)
        self.att = Parameter(torch.Tensor(2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att.view(1, -1))

    def forward(self, x, edge_index, batch):
        x = self.lin(x)
        x = self.propagate(edge_index, x=x)
        x = global_mean_pool(x,batch)
        return x

    def message(self, x_i, x_j, index):
        # Concatenate node features
        cat = torch.cat([x_i, x_j], dim=1)

        # Compute attention score
        alpha = (cat * self.att).sum(dim=1)

        # Apply non-linearity
        alpha = F.leaky_relu(alpha)

        # Normalize using softmax over neighbors
        alpha = softmax(alpha, index)

        # Weight messages
        return x_j * alpha.view(-1, 1)