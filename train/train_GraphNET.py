import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from GraphModels.GraphNET import GraphNET

dataset = torch.load("data/aqsoldb_graph_dataset.pt", weights_only=False)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = GraphNET(in_channels=dataset[0].x.shape[1], out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()


EPOCHS = 50

for epoch in range(EPOCHS):

    model.train()
    total_train_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y.view(-1, 1))
        
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_loss =  total_train_loss / len(train_loader)

    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(out, data.y.view(-1, 1))
            total_test_loss += loss.item()

    test_loss = total_test_loss / len(test_loader)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

torch.save(model,'C:\\Users\\LENOVO\\AqSolDB\\models\\GraphNET.pt')