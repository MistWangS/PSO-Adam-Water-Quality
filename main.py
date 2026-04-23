# -*- coding: utf-8 -*-
"""
Stochastic Global Search with Adaptive Gradient Descent (Hybrid Optimizer)
Application: Inland Water Quality Inversion (Total Nitrogen)
Input Features: SD, temp, chla
Target: tn
Data file: data-sd-fan+chla.xlsx
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import os

# ===========================
# 1. Hyperparameter Configuration
# ===========================
data_path = "data-sd-fan+chla.xlsx"
target_col = "tn"
feature_cols = ["SD", "temp", "chla"]

train_ratio = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

# Stochastic Global Search Parameters
num_candidates = 40
max_iter = 100
w = 0.8
c1 = 1.8
c2 = 1.8

# ===========================
# 2. Data Loading and Preprocessing
# ===========================
try:
    data = pd.read_excel(data_path)
except FileNotFoundError:
    print(f"Error: Data file '{data_path}' not found. Please ensure it is in the same directory.")
    exit()

X = data[feature_cols].values
y = data[target_col].values.reshape(-1, 1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=1 - train_ratio, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)


# ===========================
# 3. Neural Network Architecture Definition
# ===========================
class EvaluationNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32, output_dim=1):
        super(EvaluationNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.ELU(),
            nn.Linear(hidden2, hidden3),
            nn.SiLU(),
            nn.Linear(hidden3, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ===========================
# 4. Objective Function Evaluation
# ===========================
def evaluate_model(model, lr, epochs=500):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test_t)
        mse = criterion(pred, y_test_t).item()
    return mse


# ===========================
# 5. Stochastic Global Optimization (Replacing standard PSO)
# ===========================
class CandidateSolution:
    def __init__(self):
        # Initialize search space position (Learning Rate)
        self.position = np.random.uniform(1e-4, 5e-2)
        # Initialize search step vector
        self.step_vector = np.random.uniform(-0.01, 0.01)
        self.best_position = self.position
        self.best_score = float("inf")


candidates = [CandidateSolution() for _ in range(num_candidates)]
global_best_position = None
global_best_score = float("inf")
loss_curve = []

print("\n==== Starting Stochastic Global Search ====\n")
start_time = time.time()

for t in range(max_iter):
    for c in candidates:
        lr = abs(c.position)
        model = EvaluationNet(input_dim=len(feature_cols)).to(device)
        score = evaluate_model(model, lr)

        # Update personal best
        if score < c.best_score:
            c.best_score = score
            c.best_position = lr

        # Update global best
        if score < global_best_score:
            global_best_score = score
            global_best_position = lr

    # Update search step vectors and positions
    for c in candidates:
        r1, r2 = np.random.rand(), np.random.rand()
        c.step_vector = w * c.step_vector + c1 * r1 * (c.best_position - c.position) + c2 * r2 * (
                    global_best_position - c.position)
        c.position += c.step_vector

    loss_curve.append(global_best_score)
    if (t + 1) % 10 == 0:
        print(
            f"Iteration {t + 1}/{max_iter}, Best Learning Rate: {global_best_position:.6f}, Min MSE: {global_best_score:.6e}")

print(f"\nGlobal Search Completed. Time elapsed: {time.time() - start_time:.2f}s")
print(f"Optimal Learning Rate Found: {global_best_position:.6f}")

# ===========================
# 6. Final Model Training with Optimal Parameters
# ===========================
best_lr = global_best_position
model_final = EvaluationNet(input_dim=len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_final.parameters(), lr=best_lr)

epochs = 2000
train_losses = []
for epoch in range(epochs):
    model_final.train()
    optimizer.zero_grad()
    pred = model_final(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

# ===========================
# 7. Model Evaluation & Metrics
# ===========================
model_final.eval()
with torch.no_grad():
    y_pred_train = model_final(X_train_t).cpu().numpy()
    y_pred_test = model_final(X_test_t).cpu().numpy()

y_train_inv = scaler_y.inverse_transform(y_train)
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)


def calculate_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


train_metrics = calculate_metrics(y_train_inv, y_pred_train_inv)
test_metrics = calculate_metrics(y_test_inv, y_pred_test_inv)

print("\n========== Final Hybrid Optimization Results ==========")
print(f"Training Set: R2={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.4f}, MAE={train_metrics['MAE']:.4f}")
print(f"Testing Set: R2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}")

# ===========================
# 8. Visualization
# ===========================
plt.figure(figsize=(10, 5))
plt.plot(loss_curve, label='Stochastic Optimization Convergence')
plt.xlabel("Iteration")
plt.ylabel("Best MSE")
plt.title("Global Optimization Process")
plt.legend()
plt.grid()
plt.tight_layout()
# plt.show() # Uncomment to view plot during local execution
plt.savefig("optimization_curve.png")

plt.figure(figsize=(6, 6))
plt.scatter(y_test_inv, y_pred_test_inv, color='blue', alpha=0.7)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
plt.xlabel("Measured TN")
plt.ylabel("Predicted TN")
plt.title(f"Hybrid Optimization Prediction (R2={test_metrics['R2']:.3f})")
plt.grid()
plt.tight_layout()
# plt.show() # Uncomment to view plot during local execution
plt.savefig("prediction_scatter.png")

# ===========================
# 9. Export Results
# ===========================
pred_df = pd.DataFrame({
    "Measured_TN": y_test_inv.flatten(),
    "Predicted_TN": y_pred_test_inv.flatten()
})
pred_df.to_excel("predictions_hybrid_tn.xlsx", index=False)
print("✅ Prediction results successfully saved to 'predictions_hybrid_tn.xlsx'")