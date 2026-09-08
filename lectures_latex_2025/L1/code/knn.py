"""k-Nearest Neighbors from scratch and with scikit-learn.

There is no training phase: the whole model is the training set. Prediction
means finding the k closest points and letting them vote.
Run with ``uv run python lectures_latex_2025/L1/code/knn.py``.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
                    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])
y_train = np.array([0, 0, 1, 1, 0, 1])       # 0 = class A, 1 = class B
x_new = np.array([2.0, 3.0])
k = 3

# --- by hand -----------------------------------------------------------
distances = np.sqrt(np.sum((X_train - x_new) ** 2, axis=1))
nearest = np.argsort(distances)[:k]
votes = np.bincount(y_train[nearest])
prediction = votes.argmax()

print(f"new sample: {x_new}")
print(f"distances:  {np.round(distances, 3)}")
print(f"{k} nearest neighbours: indices {nearest}, labels {y_train[nearest]}")
print(f"majority vote -> class {prediction}")

# For regression the same neighbours would be averaged instead of voted on:
y_regression = np.array([1.0, 1.2, 8.5, 9.0, 0.8, 11.0])
print(f"regression variant (mean of neighbours): {y_regression[nearest].mean():.3f}")

# --- the same with scikit-learn ----------------------------------------
model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(f"scikit-learn prediction: class {model.predict([x_new])[0]}")
