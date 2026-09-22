"""Precision, recall and the F1 (Dice) score from the binary confusion matrix.

    uv run python lectures/L2/code/metrics_precision_recall_dice.py
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

true_labels = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
predictions = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])

# --- confusion matrix entries by hand ---------------------------------------
TP = np.sum((true_labels == 1) & (predictions == 1))
FP = np.sum((true_labels == 0) & (predictions == 1))
FN = np.sum((true_labels == 1) & (predictions == 0))
TN = np.sum((true_labels == 0) & (predictions == 0))

print(f"TP = {TP}  FP = {FP}  FN = {FN}  TN = {TN}")
print("confusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(true_labels, predictions))

precision = TP / (TP + FP)          # positive predictive value
recall = TP / (TP + FN)             # sensitivity, TPR
specificity = TN / (TN + FP)
npv = TN / (TN + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# F1 is the harmonic mean of precision and recall; for binary masks it is the
# Dice coefficient 2|X n Y| / (|X| + |Y|).
dice = 2 * TP / (2 * TP + FP + FN)

print(f"\nPrecision (PPV)  : {precision:.4f}  (sklearn "
      f"{precision_score(true_labels, predictions):.4f})")
print(f"Recall / Sens.   : {recall:.4f}  (sklearn "
      f"{recall_score(true_labels, predictions):.4f})")
print(f"Specificity      : {specificity:.4f}")
print(f"NPV              : {npv:.4f}")
print(f"Accuracy         : {accuracy:.4f}")
print(f"F1 = Dice        : {dice:.4f}  (sklearn "
      f"{f1_score(true_labels, predictions):.4f})")

# A model that always predicts the positive class has recall 1 but poor F1 --
# this is why a single one-sided metric is never enough.
always_positive = np.ones_like(true_labels)
print(f"\nalways-positive model: recall = "
      f"{recall_score(true_labels, always_positive):.4f}, "
      f"F1 = {f1_score(true_labels, always_positive):.4f}")
