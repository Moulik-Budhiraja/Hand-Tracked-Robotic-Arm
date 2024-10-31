import torch
import torch.nn as nn


# Define the same model architecture as the original
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


# Instantiate the model
model = DepthModel()

# Load the state dictionary into the model
model.load_state_dict(torch.load("depth_offset.pth"))

# Set the model to evaluation mode (useful if you're not training it)
model.eval()
