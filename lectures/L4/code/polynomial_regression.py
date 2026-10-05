"""Polynomial regression is ordinary linear regression on expanded features.

Shows that `np.polyfit`, a hand-built Vandermonde matrix with a pseudoinverse
and the scikit-learn pipeline `PolynomialFeatures + LinearRegression` give the
same coefficients — the model stays linear in the parameters. The degree-15
curves use x rescaled to [-1, 1], exactly as the slide "Effect of polynomial
degree and regularization" (so lambda = 100 is the lambda of the slide).

    uv run python lectures/L4/code/polynomial_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def data(n=30, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 5, n)
    y = 0.5 * x ** 3 - 3 * x ** 2 + 2 * x + rng.normal(0, 3, size=n)
    return x, y


def vandermonde(x, degree):
    """Matrix [1, x, x^2, ..., x^m] — the only thing that changes vs. regression."""
    return np.vander(x, degree + 1, increasing=True)


def main():
    x, y = data()
    degree = 3

    X = vandermonde(x, degree)
    beta = np.linalg.pinv(X) @ y                      # w = X^+ y
    print("pseudoinverse :", np.round(beta, 4))
    print("np.polyfit    :", np.round(np.polyfit(x, y, degree)[::-1], 4))

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", LinearRegression()),
    ]).fit(x.reshape(-1, 1), y)
    sklearn_beta = np.r_[pipeline["reg"].intercept_, pipeline["reg"].coef_]
    print("scikit-learn  :", np.round(sklearn_beta, 4))

    x_fit = np.linspace(0, 5, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=20, label="data")
    ax.plot(x_fit, vandermonde(x_fit, degree) @ beta, linewidth=2,
            label=f"degree {degree}")
    # Degree 15 as on the slide "Effect of polynomial degree and
    # regularization": the powers are taken of t = (x - 2.5) / 2.5 in [-1, 1]
    # (on the raw powers x^1 ... x^15 of x in [0, 5] the matrix is numerically
    # singular) and the intercept is not penalized — sklearn's Ridge does not
    # penalize it either, so lambda here is the lambda of the slide.
    def rescale(v):
        return ((np.asarray(v) - 2.5) / 2.5).reshape(-1, 1)

    for higher_degree, lam in [(15, 0.0), (15, 100.0)]:
        if lam == 0:
            model = LinearRegression()
        else:
            model = Ridge(alpha=lam)
        p = Pipeline([
            ("poly", PolynomialFeatures(degree=higher_degree, include_bias=False)),
            ("reg", model),
        ]).fit(rescale(x), y)
        y_fit = p.predict(rescale(x_fit))
        ax.plot(x_fit, y_fit, linewidth=1.5, linestyle="--",
                label=f"degree {higher_degree}, lambda={lam:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
