"""Load the leaves, split them, fit the classifier, print the accuracy.

Run it with

    python evaluate.py

The data are in `leaves.csv` next to this file, so no internet connection is
needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from knn import KNNClassifier

DATA_PATH = Path(__file__).with_name("leaves.csv")


def load_leaves(path=DATA_PATH):
    """The 518 measured leaves: heights and widths in cm, and the tree type."""
    table = pd.read_csv(path)
    X = table[["Leaf height", "Leaf width"]].to_numpy(dtype=float)
    y = table["Tree type"].to_numpy()
    return X, y


def train_test_split(X, y, test_size=0.3, seed=0):
    """Shuffle the samples and cut off `test_size` of them as the test part."""
    X = np.asarray(X)
    y = np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(X))
    n_test = int(round(test_size * len(X)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true, y_pred):
    """Share of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def main():
    X, y = load_leaves()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=0)

    # TODO: k is hardcoded and the features are not scaled.
    model = KNNClassifier(k=5).fit(X_train, y_train)

    print(f"{len(X_train)} training and {len(X_test)} test leaves")
    print(f"k = {model.k}   test accuracy = {accuracy(y_test, model.predict(X_test)):.4f}")


if __name__ == "__main__":
    main()
