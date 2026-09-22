"""Hold-out validation and k-fold cross-validation, by hand and with sklearn.

The test set is split off first and never touched again; the remaining data is
used either for a single hold-out split or for k-fold cross-validation.

    uv run python lectures/L2/code/cross_validation_holdout.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           class_sep=1.2, random_state=42)

# The test set is put aside once and used only for the final number.
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)

# --- hold-out validation (single split) -------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2,
                                                  random_state=42, stratify=y_dev)
model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
print(f"hold-out validation accuracy : "
      f"{accuracy_score(y_val, model.predict(X_val)):.4f}")

# --- k-fold cross-validation written out by hand ----------------------------
k = 5
rng = np.random.RandomState(0)
indices = rng.permutation(X_dev.shape[0])
folds = np.array_split(indices, k)

accuracies = []
for i in range(k):
    val_idx = folds[i]
    train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
    fold_model = KNeighborsClassifier(n_neighbors=3)
    fold_model.fit(X_dev[train_idx], y_dev[train_idx])
    accuracies.append(accuracy_score(y_dev[val_idx],
                                     fold_model.predict(X_dev[val_idx])))

accuracies = np.array(accuracies)
print(f"{k}-fold CV (numpy)            : {accuracies.mean():.4f} "
      f"+- {accuracies.std():.4f}")

# --- the same with scikit-learn ---------------------------------------------
kf = KFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_validate(KNeighborsClassifier(n_neighbors=3), X_dev, y_dev,
                        cv=kf, scoring="accuracy")["test_score"]
print(f"{k}-fold CV (KFold)            : {scores.mean():.4f} "
      f"+- {scores.std():.4f}")

# Stratified CV keeps the class proportions in every fold -- always prefer it
# for classification, and use GroupKFold when samples share a patient/device.
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_validate(KNeighborsClassifier(n_neighbors=3), X_dev, y_dev,
                        cv=skf, scoring="accuracy")["test_score"]
print(f"{k}-fold CV (StratifiedKFold)  : {scores.mean():.4f} "
      f"+- {scores.std():.4f}")

# --- final, one-shot evaluation on the untouched test set -------------------
final = KNeighborsClassifier(n_neighbors=3).fit(X_dev, y_dev)
print(f"test accuracy (used once)    : "
      f"{accuracy_score(y_test, final.predict(X_test)):.4f}")
