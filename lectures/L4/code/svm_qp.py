"""Hard-margin SVM solved directly as quadratic programming in the dual.

The dual from the lecture

    max_alpha  sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j x_i^T x_j
    s.t.       sum_i alpha_i y_i = 0,  alpha_i >= 0

is rewritten into the standard QP form

    min_alpha  1/2 alpha^T P alpha + q^T alpha,
    P = (y y^T) * (X X^T),  q = -1,  A alpha = 0,  alpha >= 0

The original notebook `SVM_QP.ipynb` solves it with the cvxopt library; here
`scipy.optimize.minimize` (SLSQP) is used so that the example runs without an
extra dependency.

    uv run python lectures/L4/code/svm_qp.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def data(n=50, seed=42):
    rng = np.random.RandomState(seed)
    X_pos = rng.normal(loc=[1.0, 1.0], scale=0.8, size=(n // 2, 2))
    X_neg = rng.normal(loc=[-1.2, -1.2], scale=0.8, size=(n // 2, 2))
    return np.vstack([X_pos, X_neg]), np.hstack([np.ones(n // 2),
                                                 -np.ones(n // 2)])


def solve_dual(X, y):
    n = len(y)
    P = np.outer(y, y) * (X @ X.T)

    def objective(a):
        return 0.5 * a @ P @ a - a.sum()

    def grad(a):
        return P @ a - np.ones(n)

    constraints = [{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y}]
    solution = minimize(objective, np.zeros(n), jac=grad, method="SLSQP",
                        bounds=[(0, None)] * n, constraints=constraints,
                        options={"maxiter": 500, "ftol": 1e-10})
    return solution.x


def main():
    X, y = data()
    alpha = solve_dual(X, y)

    w = ((alpha * y)[:, None] * X).sum(axis=0)
    sv = np.where(alpha > 1e-5)[0]
    b = np.mean(y[sv] - X[sv] @ w)
    print("w =", np.round(w, 4), " b =", round(float(b), 4))
    print("support vectors:", len(sv), "of", len(y))
    print("accuracy:", np.mean(np.sign(X @ w + b) == y))

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    Z = (np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.contourf(xx, yy, Z > 0, alpha=0.3)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"])
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.scatter(X[sv, 0], X[sv, 1], s=140, facecolors="none", edgecolors="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Hard-margin SVM from the dual QP")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
