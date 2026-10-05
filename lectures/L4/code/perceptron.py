"""Perceptron: training with a single rule and the boundary it learns.

Exactly the code from the slide, plus data and plotting so that it can be run:

    uv run python lectures/L4/code/perceptron.py

Both the data and the visualization layout come from the original notebook
`perceptron.ipynb`. The first column of X is a constant one, x_0 = 1 (the bias
trick, as on the slides), so the threshold w_0 = b is learned together with the
weights.
"""

import matplotlib.pyplot as plt
import numpy as np


def train_perceptron(X, y, epochs):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        for i in range(n):
            y_hat = X[i] @ w
            if y[i] * y_hat <= 0:
                w += y[i] * X[i]
    return w


def predict_perceptron(X, w):
    return np.sign(X @ w)


def train_perceptron_counting(X, y, epochs):
    """The same training, plus alpha_i: the signed number of updates per sample.

    Every update adds y_i x_i to w, which starts at zero, so at the end
    w = sum_i alpha_i x_i — the dual form from the slide.
    """
    n, d = X.shape
    w = np.zeros(d)
    alpha = np.zeros(n)
    for _ in range(epochs):
        for i in range(n):
            if y[i] * (X[i] @ w) <= 0:
                w += y[i] * X[i]
                alpha[i] += y[i]
    return w, alpha


def data(n=50, seed=42):
    rng = np.random.RandomState(seed)
    X_pos = rng.normal(loc=[2.2, 2.2], scale=0.8, size=(n // 2, 2))
    X_neg = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n // 2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])
    return np.c_[np.ones(len(X)), X], y


def main():
    X, y = data()
    w = train_perceptron(X, y, epochs=5)
    print("w =", w)
    print("training accuracy:", np.mean(predict_perceptron(X, w) == y))

    # the dual view: w is a weighted sum of the training samples
    w_again, alpha = train_perceptron_counting(X, y, epochs=5)
    print("samples with alpha_i != 0:", np.count_nonzero(alpha), "of", len(X))
    print("sum_i alpha_i x_i == w:", np.allclose(alpha @ X, w_again)
          and np.allclose(w_again, w))

    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()]
    Z = predict_perceptron(grid, w).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 1], X[:, 2], c=y)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Perceptron decision boundary")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
