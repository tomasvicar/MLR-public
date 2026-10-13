"""A k-nearest-neighbours classifier written from scratch.

Nothing here comes from scikit-learn on purpose — this is the code you wrote
yourself in exercises 1 and 2, cleaned up into a module that other files can
import.
"""

import numpy as np


def euclidean_distances(A, B):
    """Pairwise Euclidean distances between the rows of `A` and the rows of `B`.

    `A` has shape (n, d), `B` has shape (m, d), the result has shape (n, m)
    and `result[i, j]` is the distance from `A[i]` to `B[j]`.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def standardize(X):
    """Column-wise z-score: every feature gets mean 0 and standard deviation 1.

    Distances treat all features alike, so a feature measured in a larger unit
    would dominate the vote. See exercise 2 for what this does to k-NN.
    """
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


class KNNClassifier:
    """Majority vote of the `k` nearest training samples."""

    def __init__(self, k=5):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def fit(self, X, y):
        """Remember the training set. k-NN has no other training to do."""
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        if len(self.X_train_) < self.k:
            raise ValueError("fewer training samples than neighbours")
        return self

    def predict(self, X):
        """Predicted label for every row of `X`."""
        distances = euclidean_distances(X, self.X_train_)
        nearest = np.argsort(distances, axis=1)[:, : self.k]
        predictions = []
        for row in self.y_train_[nearest]:
            labels, counts = np.unique(row, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)
