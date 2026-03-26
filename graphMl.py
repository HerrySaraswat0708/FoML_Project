import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# dataset = torch.load("data/aqsoldb_graph_dataset.pt")
dataset = torch.load("data/aqsoldb_graph_dataset.pt", weights_only=False)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# print(dataset[10].y)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.lin(x)
    
model = GCN(input_dim=dataset[0].x.shape[1], hidden_dim=64)
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

# for epoch in range(1, 51):
#     train_loss = train()
#     test_loss = test()

#     print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")