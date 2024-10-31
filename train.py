import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

# Load the data from CSV
data = pd.read_csv("data.csv")

# Convert the DataFrame to a PyTorch tensor
target = torch.tensor(data["est_z"].values, dtype=torch.float32)
target = target.view(-1, 1)
features = torch.tensor(data.drop(columns=["est_z", "time"]).values, dtype=torch.float32)


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        self.layer1 = nn.Linear(63, 128)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = DepthModel()

loss_fn = nn.MSELoss()
# output = model(features)
# loss = loss_fn(output, target)
# print(loss)

if torch.cuda.is_available():
    model = model.to("cuda")
    features = features.to("cuda")
    target = target.to("cuda")

optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(15):  # Number of training epochs
    optimizer.zero_grad()
    output = model(features)  # Forward pass
    loss = loss_fn(output, target)
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    print(f"Epoch {epoch}, Loss: {loss.item()}")


torch.save(model.state_dict(), "depth_offset.pth")
