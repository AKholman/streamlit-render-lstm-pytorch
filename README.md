
# 🚀 ML-Powered Pipeline for Daily AAPL Stock Price Forecasting

This is an END-TO-END MACHINE LEARNING project that implements a complete time series forecasting pipeline. It integrates PYTORCH LSTM for sequence modeling, AIRFLOW for workflow orchestration, MLFLOW for experiment tracking, and STREAMLIT for interactive deployment.

The entire system runs locally and demonstrates a lightweight MACHINE LEARNING SYSTEM DESIGN (MLSD) — covering the full ML lifecycle from data ingestion to model deployment — without external cloud or CI/CD dependencies.  
 
---

## 📊 Models Trained and Tested
We compared multiple approaches:

| Model Type  | Library Used  | Notes |
|-------------|---------------|-------|
| SARIMA | Classical Time Series (TSA) | Baseline for seasonality patterns |
| Random Forest | scikit-learn | Ensemble regression |
| XGBoost | xgboost | Boosted decision trees |
| LSTM | TensorFlow/keras | Deep learning model |
| LSTM | PyTorch | Deep learning model – **best performing** |

The **LSTM (Long Short-Term Memory)** model implemented in **PyTorch** achieved the highest predictive performance.

---

## 🧠 LSTM Model Details (PyTorch)

### Data Scaling
All features (X) and target (y) were scaled using **Min–Max scaling**:
- Fit scalers on **training data only**
- Apply same scalers to validation and test sets

Saved as:
- `scaler_X.joblib`
- `scaler_y.joblib`

---

### Sequence Creation
```python
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])

Purpose: transform 1D time-series into overlapping sequences.
Each sequence of 60 days becomes one LSTM sample; the next day’s close price is the label.
Output: NumPy arrays (X_train_seq, y_train_seq, etc.)

----------------------

CONVERTION to TENSORS

PyTorch models operate on tensors with automatic differentiation.
We convert NumPy matrices into PyTorch tensors:
Input format: (samples, time_steps, features)
Batch size: 32
DataLoader is used for mini-batch training.


LSTM Architecture:

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

__________________

Training Workflow
Forward Pass: Input → LSTM layers → Fully connected → Output
Compute Loss: Mean Squared Error (MSE)
Backpropagation (BPTT):
Gradients flow through fc2 → fc1 → LSTM2 → LSTM1 → through all 60 time steps
Optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad() clears old gradients
    optimizer.step() updates all weights and biases
Repeat: for every mini-batch → completes one epoch
Evaluation:
        Disable gradients (torch.no_grad())
        Predict → reverse scale → compute RMSE/MAE

_______________________

⚙️ Integration of Airflow with the Project
Project Files
| File               | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| `train.py`         | Retrains the LSTM and saves best model                          |
| `mlsd_pipeline.py` | Defines Airflow DAG that runs `train.py` (scheduled or manual)  |
| `app.py`           | Streamlit app that loads the latest model and shows predictions |

            train.py  →  [Airflow DAG: mlsd_pipeline.py]  →  app.py (Streamlit)

------------------------

🧭 MLFLOW Integration

MLflow automatically tracks training runs from train.py, logging:
        Metrics (loss, MAE, RMSE)
        Hyperparameters (epochs, learning rate, etc.)
        Model artifacts (best_lstm_model.pth, scaler_X.joblib, scaler_y.joblib)
        Versioning info and run metadata
Local tracking:
        Artifacts and runs stored in /mlruns
        No server required for local experiments

We can also manually start the MLflow UI with:
            mlflow ui --backend-store-uri ./mlruns

_______________________
⚙️ Operation Modes

|      Case                | File You Run      | How It Works              | When to Use                |
| ------------------------ | ----------------- | ------------------------  | -------------------------- |
| **1️⃣ Trigger via         |                   | Airflow executes all.     |  Use for production-like.   |
|Airflow UI (automated)**  | `mlsd_pipeline.py`| steps defined in the DAG. |       automation            |
|--------------------------------------------------------------------------------------------------------|
| **2️⃣ Run manually        |                   | Direct training and MLflow |   Use for debugging        |
| (testing)**              | `train.py`        | logging without Airflow.   |   or rapid testing         |
|--------------------------------------------------------------------------------------------------------|

🗂️ Project Files Summary

| File                 | Purpose             |      Main Tasks         
| -------------------- | ------------------- | ------------------------------------------------------
| **train.py**         | Model training      | - Loads data<br
|                      |                     | - Creates sequences
|                      |                     | - Trains LSTM with PyTorch<br
|                      |                     | - Saves `best_lstm_model.pth`, `scaler_X.joblib`, 
|                      |                     |`scaler_y.joblib`
|                      |                     | - Logs everything to MLflow 

| **mlsd_pipeline.py** | Airflow DAG         | - Defines task dependencies 
|                      |                     |- Runs `train.py` and optionally starts `app.py` 
|                      |                     | Can be triggered via Airflow scheduler or manually
|                      |                     |
| **app.py**           | Streamlit Dashboard | - Loads latest model and scalers 
|                      |                     |- Displays predictions, metrics, and charts interactively
|----------------------|---------------------|-----------------------------------------------------------



🧩 Libraries and Tools Used:

| Tool / Library | Role            | Key Features           | UI               | UI Function  
| -------------- | --------------- | ---------------------- | ---------------- | ----------------------- 
| **PyTorch**    | ML framework    | LSTM model,            |   ❌             | N/A
|                |                 | tensor ops, autograd   |                  |
|                |                 |                        |                  |
| **Airflow**    | Workflow        | DAG scheduling,       | ✅ Airflow Web UI  | Visualize DAGs, triggger
|                | orchestration   | task dependencies     |                    |  tasks, monitor logs 
|                |                 |                       |                    |
| **MLflow**     | Experiment      | Logs metrics,         | ✅ MLflow UI       | Compare runs, inspect 
|                |  tracking       | parameters, and models|                    |models and metrics    
|                |                 |                       |                    |
| **Streamlit**  | Web dashboard   | Visual model results  | ✅ Streamlit Web UI | User-facing visualization 
|                |                 |  and predictions      |                    | and testing interface 
|----------------------------------------------------------------------------------------------------------


✅ Summary

 - Airflow automates and orchestrates the pipeline
 - PyTorch LSTM delivers accurate stock forecasts
 - MLflow tracks every training experiment
 - Streamlit provides an interactive dashboard
 - All components operate locally, seamlessly integrated through the Airflow DAG



🔹 Future Extensions
 - Dockerize the project for reproducibility
 - Add model registry and versioning via MLflow
 - Deploy Streamlit + Airflow on Render or AWS

_________________________________________________________________________________________________________
_________________________________________________________________________________________________________

APPENDICES: 

Machine Learning System Design with integrated AIRFLOW, MLflow, Streamlit:

1. Add small Python scripts in local_MLSD/:
    train.py — retrain the LSTM and save best_lstm_model.pth.

2. Add a DAG in Airflow’s dags/ folder that:
    Runs train.py (scheduled or manual).
    
3. MLFlow tracking experiments, metrics, models and params. 

4. Streamlit (app.py) as UI best_lstm_model.pth.



train.py --> [airflow DAG - mlst_best_model.pth] --> [app.py manually run streamlit]


             ┌─────────────────────┐
             │  Airflow Webserver  │
             │ - (UI, monitoring)  |
             | - trigger DAG       │ 
             └─────────┬──────────-┘
                       │ Shows DAGs & Task Status
                       │
             ┌─────────▼──────────----┐
             │  Airflow Scheduler     │
             │  (Runs continuously    │
             │   in background)       |
             |checks DAGs and triggers│
             └─────────┬──────----────┘
                       │
                       │ Triggers scheduled tasks
                       ▼
              ┌─────────────────--┐
              │ airflow executor: |    
              | train.py          │
              │ (LSTM model       │
              │  training &       │
              │  saving artifacts)│
              └─────────┬─────────┘
                        │
        ┌───────────────┴───────────--------────┐
        │                                       │
        ▼                                       ▼
 ┌──────────-────-------─┐               ┌───────────────------┐
 │ Best LSTM Model.      │               │ Scalers & Data.     │
 │ (best_lstm_model.pth) │               │ (scaler_X, scaler_y)│
 └───────────--------────┘               └───────────────------┘
        │                                        │
        └───────────────┬──────────────----------|           
                        |
                        |
                        ▼
                 ┌───────────────┐
                 │ Streamlit App │
                 │  (Interactive │
                 │   UI for      │
                 │  predictions) │
                 └───────────────┘
           

MLFLOW INTEGRATION:  

✅ In MLflow tracking the train.py every training run automatically logs:

    - Training metrics (loss, accuracy, etc.)
    - Parameters (hyperparameters, learning rate, epochs etc.)
    - Model artifacts (best_lstm_model.pth, scaler_X.joblib, scaler_y.joblib)
    - And versioning info

MLflow integration:
train.py logs model and metrics to the MLflow server (local folders /mlruns and /mlartifacts).
MLflow is separate from Airflow and Streamlit but can be tracked automatically whenever train.py runs.

Manual run of train.py:
You can run python train.py independently of Airflow; it will train the model and log everything to MLflow.
Streamlit will automatically pick up the latest model and scalers.

Airflow DAG (mlsd_pipeline.py) triggers tasks:
Airflow scheduler must be running for DAG-triggered tasks.
Webserver only displays the UI and allows manual DAG triggers; it does not execute tasks.


We create:                    train.py, mlsd_pipeline.py, app.py, requirements.txt
Script execution creates:     best_lstm_model.pth, scalar_X_joblib, scalar_y.joblib, mlruns (folder)  

|-------------------|
| airflow webserver |     
|   DAG, UI         |
| mlsd_pipeline.py  |                 Manual
|  orchestration    |        |----------------------------|
|-------------------|        |  train.py LSTM model.      |
         |                   | saves best_lstm_nodel.pth  | 
         V                   | saves scalarX.joblob.      |
|-------------------|        | saves scalary.joblib       |
| airflow scheduler | ---->  | log model and metrics      |
|                   |        | to MLFlow                  |
|-------------------|        |----------------------------|
         |                                 |
         |                                 |
         |                                 V
         |                   |--------------------------|       |------------------------|
         |                   |   app.py (Streamlit)     |       |   MLFlow               |
         |---------------->  | - loads models, scalers  | ....> | - tracking experiments |
                             | - shows predictions      |       | models, metrics, param |
                             | - reads update results   |       | saves in mlruns        |
                             |--------------------------|       |------------------------|    


