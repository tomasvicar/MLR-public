"""Feature expansion: a linear model becomes non-linear in the expanded space.

Port of the original Colab notebook `pomocne_colaby/feature_expansion.ipynb`.
Two concentric rings are not linearly separable in 2D; after the explicit map
Phi(x) = (x_1, x_2, x_1^2 + x_2^2) a plain logistic regression separates them.

    uv run python lectures/L3/code/feature_expansion_example.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def rings(rng, n=200):
    """Inner disc (class 0) and outer ring (class 1)."""
    r_inner = 0.6 + 0.05 * rng.randn(n)
    a_inner = 2 * np.pi * rng.rand(n)
    inner = r_inner[:, None] * np.c_[np.cos(a_inner), np.sin(a_inner)]

    r_outer = 1.4 + 0.08 * rng.randn(n)
    a_outer = 2 * np.pi * rng.rand(n)
    outer = r_outer[:, None] * np.c_[np.cos(a_outer), np.sin(a_outer)]

    X = np.vstack([inner, outer])
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def phi(X: np.ndarray) -> np.ndarray:
    """Phi(x) = (x_1, x_2, x_1^2 + x_2^2)."""
    x1, x2 = X[:, 0], X[:, 1]
    return np.stack([x1, x2, x1 ** 2 + x2 ** 2], axis=1)


def main() -> None:
    rng = np.random.RandomState(42)
    X, y = rings(rng)

    plain = LogisticRegression(max_iter=5000).fit(X, y)
    expanded = LogisticRegression(max_iter=5000).fit(phi(X), y)
    print(f"accuracy in the original 2D space: {plain.score(X, y):.3f}")
    print(f"accuracy in the expanded 3D space: {expanded.score(phi(X), y):.3f}")

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121)
    for c, name in enumerate(["class 0 (inner)", "class 1 (outer)"]):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=12, label=name)
    ax.set_aspect("equal")
    ax.set_title("Before: 2D data (not linearly separable)")
    ax.legend()

    ax = fig.add_subplot(122, projection="3d")
    Z = phi(X)
    for c in (0, 1):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], Z[y == c, 2], s=8)
    w1, w2, w3 = expanded.coef_[0]
    b = expanded.intercept_[0]
    g = np.linspace(-1.8, 1.8, 40)
    XX, YY = np.meshgrid(g, g)
    ax.plot_surface(XX, YY, -(b + w1 * XX + w2 * YY) / w3, alpha=0.25, linewidth=0)
    ax.set_title(r"Expanded space via $\Phi(x)$ (linearly separable)")
    ax.view_init(elev=18, azim=10)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
