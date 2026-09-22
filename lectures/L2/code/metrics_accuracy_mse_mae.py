"""Accuracy, MSE, MAE and RMSE from scratch and with scikit-learn.

    uv run python lectures/L2/code/metrics_accuracy_mse_mae.py
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# --- classification: accuracy ------------------------------------------------
y_true_cls = np.array([1, 0, 1, 2, 0])
y_pred_cls = np.array([1, 1, 1, 0, 0])

accuracy = np.mean(y_true_cls == y_pred_cls)
print(f"Accuracy (numpy)       : {accuracy:.4f}")
print(f"Accuracy (sklearn)     : {accuracy_score(y_true_cls, y_pred_cls):.4f}")
print(f"Error rate             : {1 - accuracy:.4f}")

# --- regression: MSE, MAE, RMSE ---------------------------------------------
y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 3.5])

mse = np.mean((y_pred_reg - y_true_reg) ** 2)
mae = np.mean(np.abs(y_pred_reg - y_true_reg))

print(f"\nMSE  (numpy)          : {mse:.4f}")
print(f"MSE  (sklearn)        : {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
print(f"MAE  (numpy)          : {mae:.4f}")
print(f"MAE  (sklearn)        : {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
print(f"RMSE (units of y)     : {np.sqrt(mse):.4f}")
