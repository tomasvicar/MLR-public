"""mRMR: minimum redundancy, maximum relevance — greedy implementation.

Port of the original Colab notebook `pomocne_colaby/mRMR.ipynb`. Relevance is
the mutual information with the label, redundancy the average mutual
information with the already selected features.

    uv run python lectures/L3/code/mrmr_example.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def mrmr_feature_selection(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    selected: list[str] = []
    remaining = list(X.columns)

    # 1) start with the feature that has the highest MI with the target
    relevance_all = pd.Series(mutual_info_classif(X, y, random_state=42),
                              index=X.columns)
    first = relevance_all.idxmax()
    selected.append(first)
    remaining.remove(first)

    # 2) then repeatedly maximize (relevance - redundancy)
    for _ in range(k - 1):
        if not remaining:
            break
        relevance = pd.Series(
            mutual_info_classif(X[remaining], y, random_state=42), index=remaining)
        redundancy = pd.Series(
            [np.mean(mutual_info_regression(X[selected], X[f], random_state=42))
             for f in remaining], index=remaining)
        scores = relevance - redundancy
        best = scores.idxmax()
        selected.append(best)
        remaining.remove(best)

    return selected


def main() -> None:
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                               n_redundant=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    selected = mrmr_feature_selection(X, y, k=5)
    print("selected by mRMR:", selected)

    print("\nplain MI with the target (for comparison):")
    for name, score in zip(X.columns, mutual_info_classif(X, y, random_state=42)):
        print(f"  {name:<12s} {score:.4f}")


if __name__ == "__main__":
    main()
