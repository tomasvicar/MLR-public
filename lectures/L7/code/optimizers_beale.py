"""A comparison of optimizers on the Beale function.

The Beale function has its minimum at (3; 0.5) and around it a very flat valley
with extremely steep walls — exactly the situation for which the lecture
introduces momentum and an adaptive step size.

All the optimizers are written from scratch following the formulas of the lecture:
SGD, momentum, NAG, AdaGrad, AdaDelta, RMSprop, Adam (bias correction included),
AdaMax and NAdam.

    MPLBACKEND=Agg uv run python lectures/L7/code/optimizers_beale.py
"""

import matplotlib.pyplot as plt
import numpy as np

MINIMUM = np.array([3.0, 0.5])
START = np.array([2.4, -2.8])


def beale(x, y):
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def gradient(point):
    x, y = point
    a = 1.5 - x + x * y
    b = 2.25 - x + x * y ** 2
    c = 2.625 - x + x * y ** 3
    return np.array([
        2 * a * (y - 1) + 2 * b * (y ** 2 - 1) + 2 * c * (y ** 3 - 1),
        2 * a * x + 4 * b * x * y + 6 * c * x * y ** 2,
    ])


def optimize(method, eta, steps=3000, beta1=0.9, beta2=0.999, rho=0.9,
             gamma=0.9, eps=1e-8):
    x = START.astype(float).copy()
    points = [x.copy()]
    m = np.zeros(2)          # the first moment / the velocity
    v = np.zeros(2)          # the second moment / the accumulated squared gradient
    delta = np.zeros(2)      # AdaDelta: the accumulated squares of the steps
    for it in range(1, steps + 1):
        g = gradient(x - gamma * m) if method == "NAG" else gradient(x)
        if method == "SGD":
            x -= eta * g
        elif method in ("Momentum", "NAG"):
            m = gamma * m + eta * g
            x -= m
        elif method == "AdaGrad":
            v += g ** 2
            x -= eta * g / (np.sqrt(v) + eps)
        elif method == "AdaDelta":
            v = rho * v + (1 - rho) * g ** 2
            step = -np.sqrt(delta + eps) / np.sqrt(v + eps) * g
            delta = rho * delta + (1 - rho) * step ** 2
            x += step
        elif method == "RMSprop":
            v = rho * v + (1 - rho) * g ** 2
            x -= eta * g / (np.sqrt(v) + eps)
        elif method in ("Adam", "NAdam"):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** it)          # bias correction
            v_hat = v / (1 - beta2 ** it)
            direction = m_hat if method == "Adam" else (
                beta1 * m_hat + (1 - beta1) * g / (1 - beta1 ** it))
            x -= eta * direction / (np.sqrt(v_hat) + eps)
        elif method == "AdaMax":
            m = beta1 * m + (1 - beta1) * g
            v = np.maximum(beta2 * v, np.abs(g))
            x -= eta / (1 - beta1 ** it) * m / (v + eps)
        else:
            raise ValueError(method)
        if not np.all(np.isfinite(x)):
            break
        points.append(x.copy())
    return np.array(points)


SETTINGS = (("SGD", 1e-5), ("Momentum", 1e-5), ("NAG", 1e-5),
            ("AdaGrad", 2.0), ("AdaDelta", 1.0), ("RMSprop", 0.02),
            ("Adam", 0.2), ("AdaMax", 0.8), ("NAdam", 0.2))


def main():
    x = np.linspace(-1.5, 4.75, 400)
    y = np.linspace(-3.0, 1.4, 400)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.contourf(X, Y, np.log10(beale(X, Y) + 1e-6), levels=40, cmap="hot")

    for method, eta in SETTINGS:
        points = optimize(method, eta)
        distance = np.linalg.norm(points[-1] - MINIMUM)
        print(f"{method:9s} eta = {eta:<7g} end = "
              f"({points[-1][0]:6.3f}, {points[-1][1]:6.3f})   "
              f"|x − x*| = {distance:7.4f}   f = {beale(*points[-1]):9.5f}")
        ax.plot(points[:, 0], points[:, 1], linewidth=1.6, label=method)

    ax.plot(*MINIMUM, marker="X", color="#00e5ff", markersize=11,
            markeredgecolor="black")
    ax.plot(*START, marker="o", color="white", markersize=7,
            markeredgecolor="black")
    ax.set(xlim=(-1.5, 4.75), ylim=(-3.0, 1.4), xlabel="x", ylabel="y",
           title="Beale function — minimum at (3, 0.5)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
