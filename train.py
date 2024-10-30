import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV data
data = pd.read_csv('training_data.csv')
PATH = "hand_depth_model.pt"

# Define the columns for the landmarks
landmark_columns = [f'world_landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

# Extract input features (landmarks) and target variable (expectedZ)
X = data[landmark_columns].values  # Shape: (num_samples, 63)
y = data['expectedZ'].values       # Shape: (num_samples,)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize the scaler for the target variable
scaler_y = StandardScaler()

# Reshape y for the scaler and transform
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Move tensors to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Helps prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output layer
        )

    def forward(self, x):
        return self.layers(x)

input_size = X_train.shape[1]  # Should be 63
model = MLP(input_size).to(device)
# model.load_state_dict(torch.load(PATH, weights_only=True))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 64

# Create DataLoader for batch processing
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss every epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

    # If target was normalized, inverse transform predictions
    predictions = scaler_y.inverse_transform(predictions.cpu().numpy())
    y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

    # Calculate additional metrics if needed, e.g., MAE
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_original, predictions)
    print(f'Mean Absolute Error: {mae:.4f}')

torch.save(model.state_dict(), PATH)