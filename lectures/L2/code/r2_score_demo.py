"""R^2 (coefficient of determination): definition, sklearn, and two fits.

    uv run python lectures/L2/code/r2_score_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- definition on a tiny example -------------------------------------------
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

sst = np.sum((y_true - np.mean(y_true)) ** 2)   # sum of squares total
sse = np.sum((y_true - y_pred) ** 2)            # sum of squares error
print(f"R^2 (numpy)   : {1 - sse / sst:.4f}")
print(f"R^2 (sklearn) : {r2_score(y_true, y_pred):.4f}")

# Always predicting the mean gives R^2 = 0.
print(f"R^2 of the mean predictor: "
      f"{r2_score(y_true, np.full_like(y_true, y_true.mean(), dtype=float)):.4f}")

# --- a good fit and a bad fit -----------------------------------------------
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 50).reshape(-1, 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, noise, title in [(axes[0], 1.0, "small noise"), (axes[1], 5.0, "large noise")]:
    y = 2 * x.ravel() + 1 + rng.normal(0, noise, size=x.shape[0])
    fit = LinearRegression().fit(x, y)
    yhat = fit.predict(x)
    ax.scatter(x, y, s=14, label="data")
    ax.plot(x, yhat, color="red", label="fitted line")
    ax.axhline(y.mean(), linestyle="--", color="gray", label="mean predictor")
    ax.set(xlabel="x", ylabel="y",
           title=f"{title}: $R^2$ = {r2_score(y, yhat):.2f}")
    ax.legend()

fig.tight_layout()
plt.show()
