"""Micro, macro and weighted averaging of multiclass metrics.

A 3-class imbalanced problem (70 / 20 / 10 %); the averages are computed from
the confusion matrix by hand and cross-checked with scikit-learn.

    uv run python lectures/L2/code/macro_micro_weighted.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                           n_redundant=1, n_classes=3, n_clusters_per_class=1,
                           weights=[0.7, 0.2, 0.1], flip_y=0, random_state=42)

model = LogisticRegression(max_iter=1000).fit(X, y)
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)          # rows = actual, cols = predicted
print("Confusion matrix:\n", cm)

TP = np.diag(cm).astype(float)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
support = cm.sum(axis=1).astype(float)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# Micro: pool all decisions first, then compute one global metric.
micro_precision = TP.sum() / (TP.sum() + FP.sum())
micro_recall = TP.sum() / (TP.sum() + FN.sum())
micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

# Macro: per-class metric first, then a plain average.
macro_f1 = f1.mean()

# Weighted macro: the average is weighted by class support.
weighted_f1 = np.sum(f1 * support) / support.sum()

print(f"\nper-class F1 : {np.round(f1, 4)}   support = {support.astype(int)}")
print(f"micro-F1     : {micro_f1:.4f}  (equals accuracy in single-label "
      "multiclass)")
print(f"macro-F1     : {macro_f1:.4f}")
print(f"weighted-F1  : {weighted_f1:.4f}")

print("\nscikit-learn classification report:")
print(classification_report(y, y_pred, digits=4))
