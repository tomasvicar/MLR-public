"""Choosing the decision threshold, and checking whether the scores are probabilities.

The classifier is a logistic regression on a synthetic dataset with a rare
positive class (prevalence 5 %). Three rules pick the operating point on the
validation part; the reliability diagram and the Brier score compare the
model's probabilities with a deliberately "overconfident" version of them —
a monotone transformation that leaves the ROC AUC untouched.

    uv run python lectures/L2/code/operating_point_calibration.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=20, n_informative=10,
                           weights=[0.95, 0.05], flip_y=0.02, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5,
                                                  stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
prob = model.predict_proba(X_val)[:, 1]      # the score of the positive class
print(f"prevalence on validation : {y_val.mean():.3f}")
print(f"ROC AUC                  : {roc_auc_score(y_val, prob):.3f}")

# --- 1. three rules for the operating point (on the VALIDATION data) --------
fpr, tpr, thresholds = roc_curve(y_val, prob)
n_pos, n_neg = y_val.sum(), len(y_val) - y_val.sum()
accuracy = (tpr * n_pos + (1 - fpr) * n_neg) / len(y_val)

rules = {
    "max accuracy    ": int(np.argmax(accuracy)),
    "max Youden J    ": int(np.argmax(tpr - fpr)),
    "sensitivity 0.95": int(np.argmax(tpr >= 0.95)),   # the first threshold that reaches it
    "default t = 0.5 ": int(np.argmin(np.abs(thresholds - 0.5))),
}
print("\nrule              threshold   TPR    FPR   accuracy")
for name, i in rules.items():
    print(f"{name}   {thresholds[i]:7.3f}   {tpr[i]:.2f}   {fpr[i]:.2f}   {accuracy[i]:.3f}")

# --- 2. calibration: the same ranking, different probabilities -------------
# Sharpening the log-odds three times is monotone: the ROC and the AUC do not
# change, the probabilities do.
logit = np.log(prob / (1 - prob))
prob_over = 1 / (1 + np.exp(-3 * logit))
print(f"\nROC AUC   model {roc_auc_score(y_val, prob):.3f}   overconfident "
      f"{roc_auc_score(y_val, prob_over):.3f}")
print(f"Brier     model {brier_score_loss(y_val, prob):.4f}   overconfident "
      f"{brier_score_loss(y_val, prob_over):.4f}")
print(f"Brier by hand: {np.mean((prob - y_val) ** 2):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].plot(thresholds[1:], tpr[1:], label="sensitivity (TPR)")
axes[0].plot(thresholds[1:], 1 - fpr[1:], label="specificity (1 - FPR)")
axes[0].plot(thresholds[1:], accuracy[1:], label="accuracy")
for name, i in rules.items():
    axes[0].axvline(thresholds[i], linestyle=":", color="gray")
    axes[0].text(thresholds[i], 0.05, name.strip(), rotation=90, fontsize=8, va="bottom")
axes[0].set(xlabel="decision threshold on the predicted probability",
            ylabel="metric", xlim=(0, 1), title="Where to put the threshold?")
axes[0].legend(loc="center right")
axes[0].grid(alpha=0.3)

for values, label in [(prob, "model"), (prob_over, "overconfident")]:
    frac_pos, mean_pred = calibration_curve(y_val, values, n_bins=10)
    axes[1].plot(mean_pred, frac_pos, marker="o",
                 label=f"{label}: Brier = {brier_score_loss(y_val, values):.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
axes[1].set(xlabel="mean predicted probability", ylabel="observed fraction of positives",
            title="Reliability diagram")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()
