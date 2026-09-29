"""Wrapper feature selection: forward selection and backward elimination.

Both search directions are run on the breast-cancer dataset with a logistic
regression scored by cross-validation, exactly the loop described on the
slides, and the result is compared with scikit-learn's
`SequentialFeatureSelector`, which implements the same greedy search.

    uv run python lectures/L3/code/forward_backward_example.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def score(X, y, subset) -> float:
    """Cross-validated accuracy of a model trained on the given feature subset."""
    if not subset:
        return 0.0
    return float(cross_val_score(model(), X[:, list(subset)], y, cv=5).mean())


def forward_selection(X, y, names, max_features=5):
    """Start with nothing, always add the feature with the best improvement."""
    selected: list[int] = []
    best = 0.0
    for _ in range(max_features):
        candidates = [(score(X, y, selected + [j]), j)
                      for j in range(X.shape[1]) if j not in selected]
        # key=... breaks ties by the first (lowest) index, the way sklearn's
        # argmax does; max() over plain tuples would pick the highest index
        value, j = max(candidates, key=lambda c: c[0])
        if value <= best:                     # no further improvement -> stop
            break
        selected.append(j)
        best = value
        print(f"  + {names[j]:<28s} CV accuracy = {best:.4f}")
    return selected, best


def backward_elimination(X, y, names, min_features=5):
    """Start with everything, always drop the feature we lose the least by."""
    selected = list(range(X.shape[1]))
    best = score(X, y, selected)
    print(f"  full set of {len(selected)} features: CV accuracy = {best:.4f}")
    while len(selected) > min_features:
        candidates = [(score(X, y, [k for k in selected if k != j]), j)
                      for j in selected]
        value, j = max(candidates, key=lambda c: c[0])   # ties -> lowest index
        selected.remove(j)
        best = value
        print(f"  - {names[j]:<28s} CV accuracy = {best:.4f}")
    return selected, best


def main() -> None:
    data = load_breast_cancer()
    # feature_names is a NumPy array -> str() keeps np.str_ out of the output
    X, y, names = data.data, data.target, [str(n) for n in data.feature_names]

    print("Forward selection:")
    forward, _ = forward_selection(X, y, names)
    print("  ->", [names[j] for j in forward])

    print("\nBackward elimination (stops at 5 features):")
    backward, _ = backward_elimination(X, y, names)
    print("  ->", [names[j] for j in backward])

    print("\nComparison with sklearn.feature_selection.SequentialFeatureSelector:")
    for direction in ("forward", "backward"):
        sfs = SequentialFeatureSelector(model(), n_features_to_select=5,
                                        direction=direction, cv=5)
        sfs.fit(X, y)
        chosen = [n for n, keep in zip(names, sfs.get_support()) if keep]
        print(f"  {direction:<9s} -> {chosen}")


if __name__ == "__main__":
    main()
