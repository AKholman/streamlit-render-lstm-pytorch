
# train.py
import os
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import mlflow
import mlflow.pytorch
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "best_lstm_model.pth")
SCALER_X_PATH = os.path.join(PROJECT_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(PROJECT_DIR, "scaler_y.joblib")

# --- MLflow Tracking URI ---
mlflow.set_tracking_uri(f"file://{os.path.join(PROJECT_DIR, 'mlruns')}")

# --- Load and preprocess data ---
ticker = "AAPL"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
df = df.rename(columns={'Adj Close': 'Adj_Close'})
df = df.sort_index()
df['Target'] = df['Adj_Close'].shift(-1)
df = df.dropna()
features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

# Train/val split
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)
train = df.iloc[:train_size]
val = df.iloc[train_size:train_size + val_size]

scaler_X = MinMaxScaler().fit(train[features])
scaler_y = MinMaxScaler().fit(train[['Target']])

X_train = scaler_X.transform(train[features])
y_train = scaler_y.transform(train[['Target']])
X_val = scaler_X.transform(val[features])
y_val = scaler_y.transform(val[['Target']])

# --- Sequence creation ---
def create_sequences(X, y=None, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        if y is not None:
            ys.append(y[i + time_steps])
    if y is not None:
        return np.array(Xs), np.array(ys)
    return np.array(Xs)

time_steps = 60
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)

device = torch.device("cpu")
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# --- LSTM Model ---
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

# --- Local MLflow setup ---
project_root = Path(__file__).resolve().parent
mlruns_path = project_root / "mlruns"

# Set tracking and artifact URIs to local folder
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.set_registry_uri(f"file://{mlruns_path}")
mlflow.set_experiment("local_mlsd_experiment")

# --- Training with MLflow logging ---

epochs = 5
best_val = float('inf')

with mlflow.start_run(run_name="LSTM_Training_Run") as run:
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("input_dim", len(features))
    mlflow.log_param("hidden_dim", 64)
    mlflow.log_param("num_layers", 2)
    mlflow.log_param("time_steps", time_steps)

    model = LSTMModel(input_dim=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        losses = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        print(f"Epoch {epoch+1}/{epochs} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}")
        mlflow.log_metric("train_loss", np.mean(losses), step=epoch+1)
        mlflow.log_metric("val_loss", val_loss, step=epoch+1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=None)
            joblib.dump(scaler_X, SCALER_X_PATH)
            joblib.dump(scaler_y, SCALER_Y_PATH)
            mlflow.log_artifact(SCALER_X_PATH)
            mlflow.log_artifact(SCALER_Y_PATH)

print("✅ Training finished and logged to MLflow!")


'''# train.py
# Minimal retrain script adapted from your notebook
import os
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "best_lstm_model.pth")

# --- Data load and processing (same as your notebook) ---
ticker = "AAPL"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
df = df.rename(columns={'Adj Close': 'Adj_Close'})
df = df.sort_index()
df['Target'] = df['Adj_Close'].shift(-1)
df = df.dropna()
features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

# Train/val/test split
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)
train = df.iloc[:train_size]
val = df.iloc[train_size:train_size + val_size]
test = df.iloc[train_size + val_size:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train[features])
y_train = scaler_y.fit_transform(train[['Target']])

X_val = scaler_X.transform(val[features])
y_val = scaler_y.transform(val[['Target']])

def create_sequences(X, y=None, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        if y is not None:
            ys.append(y[i + time_steps])
    if y is not None:
        return np.array(Xs), np.array(ys)
    return np.array(Xs)

time_steps = 60
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)

device = torch.device("cpu")  # use "cuda" only if available & desired

X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# --- Model definition (same as notebook) ---
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

model = LSTMModel(input_dim=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Train loop (small number of epochs for demo; adjust for real) ---
epochs = 5
best_val = float('inf')

for epoch in range(epochs):
    model.train()
    losses = []
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    print(f"Epoch {epoch+1}/{epochs} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_PATH)

# Save scalers so Streamlit or predict.py can invert transforms (optional)
import joblib
joblib.dump(scaler_X, os.path.join(PROJECT_DIR, "scaler_X.joblib"))
joblib.dump(scaler_y, os.path.join(PROJECT_DIR, "scaler_y.joblib"))

print("Training finished. Model saved to", MODEL_PATH)

'''