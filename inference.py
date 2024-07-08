import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
# Define the Transformer model class
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

# Load model and scaler
model = TransformerModel(feature_size=9)  
model.load_state_dict(torch.load("gold_price_predictor_epoch.pth"))
model.eval()
scaler = joblib.load("scaler.pkl")

# Load and preprocess new data
new_data_path = r"Stock price using perceptrons\unseen_dataset.csv"
new_data = pd.read_csv(new_data_path)

new_data['Date'] = pd.to_datetime(new_data['Date'], format='%d/%m/%Y')
new_data = new_data.dropna().sort_values(by='Date').reset_index(drop=True)
dates = new_data['Date'].values
features = new_data.drop(columns=['Date'])
scaled_features = scaler.transform(features)

# Create sequences
sequence_length = 5  # Previous week (5 trading days)
X_new, prediction_dates = [], []
for i in range(len(scaled_features) - sequence_length):
    X_new.append(scaled_features[i:i + sequence_length])
    prediction_dates.append(dates[i + sequence_length])

X_new = torch.tensor(X_new, dtype=torch.float32)

# Inference
with torch.no_grad():
    predictions = model(X_new).squeeze().numpy()

# Inverse transform predictions
predictions = predictions.reshape(-1, 1)
expanded_predictions = np.zeros((predictions.shape[0], scaler.n_features_in_))
expanded_predictions[:, 0] = predictions[:, 0]
original_scale_predictions = scaler.inverse_transform(expanded_predictions)[:, 0]


# Get the last day price from the scaled features
last_day_scaled = scaled_features[-2, 0]  # Assuming the first column is the price
last_day_price = scaler.inverse_transform([[last_day_scaled] + [0] * (scaler.n_features_in_ - 1)])[0, 0]

# Output predictions with dates and buy/sell indicator
for date, prediction in zip(prediction_dates, original_scale_predictions):
    action = "Buy" if prediction > last_day_price else "Sell"
    print(f"Date: {pd.to_datetime(date).strftime('%Y-%m-%d')}, Prediction: {prediction}, Last Day Price: {last_day_price}, Action: {action}")


# Output predictions with dates and buy/sell indicator
for date, prediction in zip(prediction_dates, original_scale_predictions):
    action = "Buy" if prediction > last_day_price else "Sell"
    print(f"Date: {pd.to_datetime(date).strftime('%Y-%m-%d')}, Prediction: {prediction}, Last Day Price: {last_day_price}, Action: {action}")
