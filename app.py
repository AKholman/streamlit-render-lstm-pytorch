# app.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import altair as alt

# -----------------------
# LSTM Model Definition
# -----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# -----------------------
# Helper Functions
# -----------------------
def create_sequences(X, time_steps=60):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

def load_model(model_path, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# -----------------------
# Streamlit App
# -----------------------
st.title("LSTM Stock Price Prediction")
st.write("Predict next-day stock price using a pre-trained LSTM model.")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
predict_button = st.button("Predict")

if predict_button:
    # Load historical data
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
    df = df.rename(columns={'Adj Close': 'Adj_Close'})
    df = df.sort_index()
    df['Target'] = df['Adj_Close'].shift(-1)
    df = df.dropna()
    
    features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    
    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_all = scaler_X.fit_transform(df[features])
    y_all = scaler_y.fit_transform(df[['Target']])
    
    # Create sequences
    time_steps = 60
    X_seq = create_sequences(X_all, time_steps)
    y_seq = y_all[time_steps:]
    
    # Convert to tensor
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    
    # Load model
    model, device = load_model("best_lstm_model.pth", input_dim=len(features))
    X_t = X_t.to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_t).cpu().numpy()
    
    # Inverse transform
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)
    
    # Show last actual vs predicted
    last_actual = df['Target'].values[-1]
    next_day_pred = y_pred_inv[-1][0]
    
    st.write(f"Last actual closing price: **${last_actual:.2f}**")
    st.write(f"Next-day predicted closing price: **${next_day_pred:.2f}**")
    
    # -----------------------
    # Altair Interactive Chart
    # -----------------------
    chart_data = pd.DataFrame({
        "Date": df.index[-30:],
        "Actual": df['Target'].values[-30:],
        "Predicted": y_pred_inv[-30:].flatten()
    })
    
    chart_data_melt = chart_data.melt('Date', var_name='Type', value_name='Price')
    
    chart = alt.Chart(chart_data_melt).mark_line().encode(
        x='Date:T',
        y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color='Type:N'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
