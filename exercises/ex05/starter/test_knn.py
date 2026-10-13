"""Tests for the classifier and the evaluation helpers.

Run them with

    pytest

They use small hand-made arrays, not the leaves file, so they need no
internet connection and finish instantly.
"""

import numpy as np

from evaluate import accuracy, train_test_split
from knn import KNNClassifier, euclidean_distances, standardize


def test_euclidean_distances_matches_the_3_4_5_triangle():
    d = euclidean_distances([[0.0, 0.0]], [[3.0, 4.0], [0.0, 0.0]])
    assert d.shape == (1, 2)
    assert np.allclose(d, [[5.0, 0.0]])


def test_one_neighbour_reproduces_the_training_labels():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    y = np.array(["a", "a", "b", "b"])
    model = KNNClassifier(k=1).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_the_vote_outweighs_a_single_odd_neighbour():
    # The closest training sample is "cherry", the next three are "walnut":
    # k = 1 follows the outlier, k = 3 overrules it (exercise 2, task 1).
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["cherry", "walnut", "walnut", "walnut"])
    new = np.array([[0.1]])
    assert KNNClassifier(k=1).fit(X, y).predict(new)[0] == "cherry"
    assert KNNClassifier(k=3).fit(X, y).predict(new)[0] == "walnut"


def test_standardize_gives_zero_mean_and_unit_std():
    X = np.array([[1.0, 100.0], [2.0, 300.0], [3.0, 200.0], [4.0, 400.0]])
    Z = standardize(X)
    assert np.allclose(Z.mean(axis=0), 0.0)
    assert np.allclose(Z.std(axis=0), 1.0)


def test_split_keeps_every_sample_exactly_once():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.array(list("aabbaabbaa"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=0)
    assert len(X_test) == 3 and len(X_train) == 7
    assert len(y_test) == 3 and len(y_train) == 7
    together = np.vstack([X_train, X_test])
    assert np.array_equal(np.sort(together[:, 0]), np.sort(X[:, 0]))


def test_accuracy_counts_the_hits():
    assert accuracy(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == 0.75
