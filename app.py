# app.py
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "best_lstm_model.pth")
SCALER_X_PATH = os.path.join(PROJECT_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(PROJECT_DIR, "scaler_y.joblib")

# --- Load or create scalers ---
features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
try:
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
except FileNotFoundError:
    st.warning("Scalers not found. Creating new ones (will be fitted on 1y AAPL data).")
    df = yf.download("AAPL", period="1y", interval="1d", auto_adjust=False)
    df = df.rename(columns={'Adj Close': 'Adj_Close'})
    df = df.sort_index()
    df['Target'] = df['Adj_Close'].shift(-1)
    df = df.dropna()
    scaler_X = MinMaxScaler().fit(df[features])
    scaler_y = MinMaxScaler().fit(df[['Target']])
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

# --- Load LSTM model ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)

device = torch.device("cpu")
model = LSTMModel(input_dim=len(features))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Streamlit UI ---
st.title("AAPL Stock Price Prediction")
st.write("Next-day prediction using trained LSTM model.")

# --- Fetch recent data ---

df = yf.download("AAPL", period="1y", interval="1d", auto_adjust=False)

if df.empty:
    st.error("❌ Failed to fetch stock data from Yahoo Finance.")
    st.stop()

'''
df = yf.download("AAPL", period="1y", interval="1d", auto_adjust=False)
'''

df = df.rename(columns={'Adj Close': 'Adj_Close'})
df = df.sort_index()
df['Target'] = df['Adj_Close'].shift(-1)
df = df.dropna()

# --- Prepare sequences ---
def create_sequences(X, time_steps=60):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

time_steps = 60
X_all = scaler_X.transform(df[features])
X_seq = create_sequences(X_all, time_steps)
X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)

# --- Predict button ---
if st.button("Predict"):
    with torch.no_grad():
        preds_scaled = model(X_t).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled)

    # Last prediction
    next_day_pred = preds[-1][0]
    st.metric(label="Predicted Next-Day AAPL Price", value=f"${next_day_pred:.2f}")

    # Combine actual and predicted for plotting
    pred_dates = df.index[time_steps:]
    plot_df = pd.DataFrame({
        "Actual": df['Target'][time_steps:].values,
        "Predicted": preds.flatten()
    }, index=pred_dates)

    st.line_chart(plot_df)
