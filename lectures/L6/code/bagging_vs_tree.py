"""Bagging reduces the variance of the prediction, not its bias.

We draw many independent training sets from the same distribution, fit a single
deep regression tree on each of them and also a bagged ensemble, and then look
at how much the prediction at a fixed point x_new jumps around. That spread is
exactly the variance term of the bias-variance decomposition.

    uv run python lectures/L6/code/bagging_vs_tree.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

N = 60              # samples in one training set
REPEATS = 200     # how many independent training sets
M = 50              # trees in the bagged ensemble
X_NEW = 0.35


def truth(x):
    return np.sin(2.6 * np.pi * x) * 0.45 + 0.55 * x + 0.5


def data(generator):
    x = generator.uniform(0, 1, N)
    return x.reshape(-1, 1), truth(x) + generator.normal(scale=0.12, size=N)


def main() -> None:
    generator = np.random.default_rng(0)
    grid = np.linspace(0, 1, 300).reshape(-1, 1)
    pred_tree, pred_bag = [], []

    for _ in range(REPEATS):
        X, y = data(generator)
        tree = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)
        bag = BaggingRegressor(DecisionTreeRegressor(max_depth=None),
                               n_estimators=M, random_state=0).fit(X, y)
        pred_tree.append(tree.predict(grid))
        pred_bag.append(bag.predict(grid))

    pred_tree = np.array(pred_tree)
    pred_bag = np.array(pred_bag)
    i = int(np.argmin(np.abs(grid.ravel() - X_NEW)))

    for name, pred in (("single tree", pred_tree), ("bagging", pred_bag)):
        variance = pred[:, i].var()
        bias = pred[:, i].mean() - truth(X_NEW)
        print(f"{name:>12}:  variance at x_new = {variance:.4f},  "
              f"bias = {bias:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, pred) in zip(axes, (("single tree", pred_tree),
                                       ("bagging (M = 50)", pred_bag))):
        for trace in pred[:40]:
            ax.plot(grid, trace, color="tab:orange", linewidth=0.6, alpha=0.25)
        ax.plot(grid, pred.mean(axis=0), color="tab:orange", linewidth=2.2,
                label="mean prediction")
        ax.plot(grid, truth(grid.ravel()), color="tab:red", linewidth=2.2,
                label="ground truth")
        ax.axvline(X_NEW, color="tab:green", linestyle="--")
        ax.set(xlabel="x", title=f"{name}: var = {pred[:, i].var():.4f}")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
