"""Mutual information as a filter feature-selection score (scikit-learn).

Port of the original Colab notebook `pomocne_colaby/MI.ipynb`:
`mutual_info_classif` for a discrete target, `mutual_info_regression` for a
continuous one, plus the pairwise MI matrix used to judge redundancy.

    uv run python lectures/L3/code/mutual_information_example.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def discrete_example() -> None:
    """Estimate MI from a count table, then check the discrete sklearn route."""
    counts = np.array([[4, 1], [1, 4]])
    joint = counts / counts.sum()
    independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    occupied = joint > 0
    mi_bits = np.sum(joint[occupied] * np.log2(joint[occupied] / independent[occupied]))

    # Reconstruct the ten observed pairs for sklearn's feature/target interface.
    pairs = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), counts.ravel(), axis=0)
    mi_nats = mutual_info_classif(
        pairs[:, :1], pairs[:, 1], discrete_features=True, random_state=42)[0]
    np.testing.assert_allclose(mi_bits, 0.2780719051126377, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mi_nats / np.log(2), mi_bits, rtol=0, atol=1e-12)
    print("Discrete pair counts:\n", counts)
    print(f"MI: {mi_bits:.7f} bits = {mi_nats:.7f} nats\n")


def main() -> None:
    discrete_example()
    iris = load_iris()
    X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_iris = pd.Series(iris.target, name="target")

    housing = fetch_california_housing()
    X_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
    y_housing = pd.Series(housing.target, name="target")

    # Continuous features + discrete target -> mixed nearest-neighbour estimator.
    mi_iris = mutual_info_classif(
        X_iris, y_iris, discrete_features=False, random_state=42)
    print("MI with the label (Iris, classification; nats):")
    for name, score in zip(X_iris.columns, mi_iris):
        print(f"  {name:<20s} {score:.4f}")

    # continuous target -> mutual_info_regression (kNN estimator)
    mi_housing = mutual_info_regression(
        X_housing, y_housing, discrete_features=False, random_state=42)
    print("\nMI with the label (California housing, regression; nats):")
    for name, score in zip(X_housing.columns, mi_housing):
        print(f"  {name:<20s} {score:.4f}")

    # pairwise MI between features = redundancy
    d = X_iris.shape[1]
    mi_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            score = mutual_info_regression(
                X_iris.iloc[:, i].values.reshape(-1, 1), X_iris.iloc[:, j],
                random_state=42)[0]
            mi_matrix[i, j] = mi_matrix[j, i] = score

    print("\nPairwise MI between Iris features (redundancy; nats; diagonal omitted):")
    print(pd.DataFrame(mi_matrix, index=X_iris.columns, columns=X_iris.columns)
          .round(3).to_string())


if __name__ == "__main__":
    main()
