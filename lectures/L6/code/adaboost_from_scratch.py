"""AdaBoost (AdaBoostM1) written from scratch, with decision stumps.

The whole algorithm is the five steps from the lecture slide: uniform weights,
pick the stump with the smallest weighted error, alpha = 0.5*ln((1-eps)/eps),
reweight, repeat. Running it on the ten points of the lecture example
reproduces exactly the numbers on the slides:

    m=1  eps=0.30  alpha=0.42
    m=2  eps=0.21  alpha=0.65
    m=3  eps=0.14  alpha=0.92

    uv run python lectures/L6/code/adaboost_from_scratch.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# the ten points from the example on the slides
X_EXAMPLE = np.array([
    [0.37, 0.77], [0.51, 0.85], [0.66, 0.72], [0.86, 0.81], [0.11, 0.52],
    [0.35, 0.57], [0.52, 0.50], [0.18, 0.29], [0.31, 0.14], [0.83, 0.23],
])
Y_EXAMPLE = np.array([+1, +1, +1, -1, +1, -1, -1, +1, -1, -1])


def stump_predict(X, feature, threshold, polarity):
    """Decision stump: +1 on one side of the threshold, -1 on the other."""
    return polarity * np.where(X[:, feature] < threshold, 1, -1)


def best_stump(X, y, weights):
    """The stump with the smallest weighted error — brute force over features and thresholds."""
    best, smallest_error = None, np.inf
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        thresholds = np.r_[values[0] - 1e-6, (values[:-1] + values[1:]) / 2]
        for threshold in thresholds:
            for polarity in (+1, -1):
                error = weights[stump_predict(X, feature, threshold, polarity) != y].sum()
                if error < smallest_error:
                    best, smallest_error = (feature, threshold, polarity), error
    return best, float(smallest_error)


def adaboost(X, y, n_rounds, verbose=False):
    """Returns a list of (stump, alpha) and the history of the weights."""
    weights = np.full(y.size, 1 / y.size)
    model, weight_history = [], [weights.copy()]
    for m in range(1, n_rounds + 1):
        stump, epsilon = best_stump(X, y, weights)
        epsilon = min(max(epsilon, 1e-10), 1 - 1e-10)
        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        if verbose:
            print(f"  m={m}  eps={epsilon:.2f}  alpha={alpha:.2f}  "
                  f"stump: x{stump[0] + 1} {'<' if stump[2] > 0 else '>='}"
                  f" {stump[1]:.3f}")
        weights = weights * np.exp(-alpha * y * stump_predict(X, *stump))
        weights /= weights.sum()                # the normalization constant Z_m
        model.append((stump, alpha))
        weight_history.append(weights.copy())
    return model, weight_history


def predict(model, X):
    score = sum(alpha * stump_predict(X, *stump) for stump, alpha in model)
    return np.sign(score)


def main() -> None:
    print("lecture example (10 points, M = 3):")
    model, history = adaboost(X_EXAMPLE, Y_EXAMPLE, 3, verbose=True)
    print("  weights w^(2):", np.round(history[1], 2))
    print("  weights w^(3):", np.round(history[2], 2))
    print("  training error of H:",
          float((predict(model, X_EXAMPLE) != Y_EXAMPLE).mean()))

    print("\nbreast cancer dataset:")
    data = load_breast_cancer()
    y = np.where(data.target == 1, 1, -1)
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, y, test_size=0.3,
                                              random_state=0, stratify=y)
    curve_tr, curve_te = [], []
    model = []
    weights = np.full(y_tr.size, 1 / y_tr.size)
    for t in range(1, 41):
        stump, epsilon = best_stump(X_tr, y_tr, weights)
        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        weights = weights * np.exp(-alpha * y_tr * stump_predict(X_tr, *stump))
        weights /= weights.sum()
        model.append((stump, alpha))
        curve_tr.append((predict(model, X_tr) != y_tr).mean())
        curve_te.append((predict(model, X_te) != y_te).mean())
    print(f"  M=1  test error {curve_te[0]:.4f}")
    print(f"  M=40 test error {curve_te[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 41), curve_tr, label="training error")
    ax.plot(range(1, 41), curve_te, label="test error")
    ax.set(xlabel="number of weak classifiers M", ylabel="error",
           title="AdaBoost with decision stumps")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
