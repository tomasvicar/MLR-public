"""ROC and precision-recall curves computed by hand and with scikit-learn.

The classifier is a logistic regression on a synthetic dataset; the curves are
built by sweeping every unique score as a threshold.

    uv run python lectures/L2/code/roc_pr_curves.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]

# --- sweep the thresholds by hand -------------------------------------------
thresholds = np.sort(np.unique(scores))[::-1]
fpr_list, tpr_list, precision_list, recall_list = [], [], [], []

for threshold in thresholds:
    predicted = (scores >= threshold).astype(int)
    tp = np.sum((predicted == 1) & (y_test == 1))
    fp = np.sum((predicted == 1) & (y_test == 0))
    tn = np.sum((predicted == 0) & (y_test == 0))
    fn = np.sum((predicted == 0) & (y_test == 1))

    tpr_list.append(tp / (tp + fn))
    fpr_list.append(fp / (fp + tn))
    precision_list.append(tp / max(tp + fp, 1))
    recall_list.append(tp / (tp + fn))

fpr_np = np.array(fpr_list)
tpr_np = np.array(tpr_list)
precision_np = np.array(precision_list)
recall_np = np.array(recall_list)

print(f"ROC AUC (numpy)   : {np.trapezoid(tpr_np, fpr_np):.4f}")
print(f"ROC AUC (sklearn) : {roc_auc_score(y_test, scores):.4f}")

# --- the same with scikit-learn ---------------------------------------------
fpr_sk, tpr_sk, _ = roc_curve(y_test, scores)
precision_sk, recall_sk, _ = precision_recall_curve(y_test, scores)
print(f"PR  AUC (sklearn) : {auc(recall_sk, precision_sk):.4f}")
print(f"prevalence        : {y_test.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot(fpr_np, tpr_np, label="numpy")
axes[0].plot(fpr_sk, tpr_sk, "--", label="sklearn")
axes[0].plot([0, 1], [0, 1], ":", color="gray", label="random")
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(recall_np, precision_np, label="numpy")
axes[1].plot(recall_sk, precision_sk, "--", label="sklearn")
axes[1].axhline(y_test.mean(), linestyle=":", color="gray", label="prevalence")
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()
