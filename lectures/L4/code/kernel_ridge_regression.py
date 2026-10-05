"""Ridge regression in three forms: primal, dual and kernel (KRR).

The primal and the dual give the very same prediction on the same data — they
differ only in whether a d x d or an n x n matrix gets inverted. The kernel
version then replaces X X^T by the matrix K and can capture nonlinear
dependencies without the features ever being computed explicitly.

    uv run python lectures/L4/code/kernel_ridge_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np


def primal(X, y, lam):
    """w = (X^T X + lambda I)^-1 X^T y"""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def dual(X, y, lam):
    """alpha = (X X^T + lambda I)^-1 y,  w = X^T alpha"""
    n = X.shape[0]
    return np.linalg.solve(X @ X.T + lam * np.eye(n), y)


def rbf_kernel(A, B, gamma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


def kernel_ridge(K, y, lam):
    """alpha = (K + lambda I)^-1 y"""
    return np.linalg.solve(K + lam * np.eye(len(y)), y)


def main():
    rng = np.random.RandomState(0)
    n, d, lam = 60, 4, 1.0
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=n)
    X_new = rng.normal(size=(10, d))

    w = primal(X, y, lam)
    alpha = dual(X, y, lam)
    print("||w_primal - X^T alpha|| =", np.linalg.norm(w - X.T @ alpha))
    print("max |y_primal - y_dual|  =",
          np.abs(X_new @ w - X_new @ X.T @ alpha).max())

    # The linear kernel is the dot product -> KRR has to match the dual.
    alpha_k = kernel_ridge(X @ X.T, y, lam)
    print("max |y_dual - y_kernel|  =",
          np.abs(X_new @ X.T @ alpha - (X_new @ X.T) @ alpha_k).max())

    # RBF kernel: nonlinear data, no explicit features.
    x = np.linspace(-3, 3, 60)
    yy = np.sin(x) + 0.15 * rng.normal(size=x.size)
    Xa = x.reshape(-1, 1)
    gamma, lam_rbf = 0.5, 0.1
    alpha_rbf = kernel_ridge(rbf_kernel(Xa, Xa, gamma), yy, lam_rbf)

    x_fit = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    y_fit = rbf_kernel(x_fit, Xa, gamma) @ alpha_rbf

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, yy, s=20, label="data")
    ax.plot(x_fit, y_fit, linewidth=2, label="KRR, RBF kernel")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
