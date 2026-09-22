"""Standardization, min-max and robust scaling -- always fitted on train only.

    uv run python lectures/L2/code/scaling_normalization.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)
# Give the two features very different scales, as real measurements have.
X = X * np.array([1.0, 50.0]) + np.array([10.0, 500.0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# --- scikit-learn: fit on train, transform both -----------------------------
for name, scaler in [("standard", StandardScaler()),
                      ("min-max ", MinMaxScaler()),
                      ("robust  ", RobustScaler())]:
    scaler.fit(X_train)                       # statistics from training data only
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)    # the *same* statistics
    print(f"{name}: train mean = {train_scaled.mean(axis=0).round(3)}, "
          f"test mean = {test_scaled.mean(axis=0).round(3)}")

# --- the same three formulas written out ------------------------------------
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_std = (X_train - mean) / std
X_test_std = (X_test - mean) / std            # training statistics, not test ones

lo, hi = X_train.min(axis=0), X_train.max(axis=0)
X_train_minmax = (X_train - lo) / (hi - lo)
X_test_minmax = (X_test - lo) / (hi - lo)

median = np.median(X_train, axis=0)
iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
X_train_robust = (X_train - median) / iqr
X_test_robust = (X_test - median) / iqr

print(f"\nz-score  train std     : {X_train_std.std(axis=0).round(3)}")
print(f"min-max  train range   : {X_train_minmax.min(axis=0).round(3)} .. "
      f"{X_train_minmax.max(axis=0).round(3)}")
print(f"robust   train median  : {np.median(X_train_robust, axis=0).round(3)}")

# A more robust variant of min-max: clip to percentiles instead of min/max.
lo_p, hi_p = np.percentile(X_train, [1, 99], axis=0)
X_test_percentile = np.clip((X_test - lo_p) / (hi_p - lo_p), 0, 1)
print(f"percentile min-max test range : {X_test_percentile.min(axis=0).round(3)}"
      f" .. {X_test_percentile.max(axis=0).round(3)}")
