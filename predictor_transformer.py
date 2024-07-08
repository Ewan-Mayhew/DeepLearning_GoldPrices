import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load and preprocess the training data
file_path = r"C:\Projects\Stock price using perceptrons\cleaned_merged_data.csv"
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data = data.dropna().sort_values(by='Date').reset_index(drop=True)
features = data.drop(columns=['Date'])

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Create sequences with the previous week's data
sequence_length = 5  # Assuming 5 trading days in a week
X, y = [], []
for i in range(len(scaled_features) - sequence_length):
    X.append(scaled_features[i:i + sequence_length])
    y.append(scaled_features[i + sequence_length][0])

X, y = np.array(X), np.array(y)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, feature_size, num_layers=6, num_heads=9, dff=128):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dim_feedforward=dff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(feature_size, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

model = TransformerModel(feature_size=X.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Load previous weights if available
try:
    model.load_state_dict(torch.load("gold_price_predictor_epoch.pth"))
    print("Loaded previous model weights.")
except FileNotFoundError:
    print("No previous weights found, training from scratch.")

# Training loop
epochs = 500
for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    
    # Save model weights after each epoch
    torch.save(model.state_dict(), f"gold_price_predictor_epoch.pth")
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save final model and scaler
torch.save(model.state_dict(), "gold_price_predictor.pth")
joblib.dump(scaler, "scaler.pkl")
