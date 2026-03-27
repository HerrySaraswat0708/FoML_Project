import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from GraphCN import GCNConv
dataset = torch.load("data/aqsoldb_graph_dataset.pt", weights_only=False)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# print(dataset[10].y)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()

#         self.lin1 = torch.nn.Linear(input_dim,hidden_dim)

#         self.conv1 = GCNConv(hidden_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.conv4 = GCNConv(hidden_dim, hidden_dim)
        
#         self.lin2 = torch.nn.Linear(hidden_dim, 1)

#     def forward(self, x, edge_index, batch):
#         x = self.lin1(x)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
        
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         x = self.conv3(x, edge_index)
#         x = F.relu(x)

#         x = self.conv4(x, edge_index)
#         x = F.relu(x)

#         x = global_mean_pool(x, batch)

#         return self.lin(x)


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.lin = Linear(in_channels, out_channels, bias=False)
#         self.bias = Parameter(torch.empty(out_channels))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.bias.data.zero_()

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         out = self.propagate(edge_index, x=x, norm=norm)

#         # Step 6: Apply a final bias vector.
#         out = out + self.bias

#         return out

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]
#         return norm.view(-1, 1) * x_j
    

model = GCNConv(in_channels=dataset[0].x.shape[1], out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        # loss = loss_fn(out, data.y)
        loss = loss_fn(out, data.y.view(-1, 1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def test():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            # loss = loss_fn(out, data.y)
            loss = loss_fn(out, data.y.view(-1, 1))
            total_loss += loss.item()

    return total_loss / len(test_loader)

for epoch in range(1, 81):
    train_loss = train()
    test_loss = test()

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")