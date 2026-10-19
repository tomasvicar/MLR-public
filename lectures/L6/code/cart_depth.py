"""CART on real data: how the depth of the tree shapes the decision boundary.

Two features of the wine dataset, one classification tree per depth. The deeper
the tree, the more rectangular pieces the boundary has — and the wider the gap
between the training and the test accuracy (overfitting).

    uv run python lectures/L6/code/cart_depth.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DEPTHS = [1, 2, 4, None]


def main() -> None:
    data = load_wine()
    features = [data.feature_names.index("flavanoids"),
                data.feature_names.index("color_intensity")]
    X, y = data.data[:, features], data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35,
                                              random_state=0, stratify=y)

    x1 = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 400)
    x2 = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.column_stack([X1.ravel(), X2.ravel()])

    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(4 * len(DEPTHS), 3.6))
    for ax, depth in zip(axes, DEPTHS):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
        tree.fit(X_tr, y_tr)
        acc_tr = tree.score(X_tr, y_tr)
        acc_te = tree.score(X_te, y_te)
        print(f"depth={str(depth):>4}  leaves={tree.get_n_leaves():>3}  "
              f"train={acc_tr:.3f}  test={acc_te:.3f}")

        ax.contourf(X1, X2, tree.predict(grid).reshape(X1.shape),
                    levels=[-0.5, 0.5, 1.5, 2.5], cmap="Pastel1")
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="Set1", s=22,
                   edgecolor="white", linewidth=0.5)
        ax.set(xlabel=data.feature_names[features[0]],
               ylabel=data.feature_names[features[1]],
               title=f"max_depth={depth}, test acc. {acc_te:.2f}")
    fig.tight_layout()

    # the tree itself is a readable model — we plot the smallest one
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
    plot_tree(tree, ax=ax2, filled=True, impurity=True, proportion=False,
              feature_names=[data.feature_names[i] for i in features],
              class_names=list(data.target_names))
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
