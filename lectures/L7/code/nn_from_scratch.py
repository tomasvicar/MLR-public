"""A neural network and backpropagation from scratch in numpy.

Exactly following the algorithm from the slide "General neural net — forward and
backward pass":

    h_0 = x
    forward:   z_k = W_k h_{k-1},        h_k = f(z_k)          k = 1 … K
    backward:  ∇_{z_k}L = ∇_{h_k}L ⊙ f'(z_k)
               ∇_{h_{k-1}}L = W_k^⊤ ∇_{z_k}L
               ∇_{W_k}L = ∇_{z_k}L h_{k-1}^⊤
    update:    W_k^(t+1) = W_k^(t) − η ∇_{W_k} L

The script first verifies the analytical gradients against a numerical derivative,
then trains a small network on two half-moons and plots the decision boundary and
the course of the loss.

    MPLBACKEND=Agg uv run python lectures/L7/code/nn_from_scratch.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def init_weights(sizes, generator):
    """Layer weights; the last column of every matrix is the bias (input x_0 = 1)."""
    return [generator.normal(scale=np.sqrt(2.0 / n_in), size=(n_out, n_in + 1))
            for n_in, n_out in zip(sizes[:-1], sizes[1:])]


def forward(weights, X):
    """Returns the list of activations h_k and of pre-activations z_k."""
    h = [X.T]
    z = []
    for k, W in enumerate(weights):
        h_with_bias = np.vstack([h[-1], np.ones((1, h[-1].shape[1]))])
        z.append(W @ h_with_bias)
        h.append(sigmoid(z[-1]))
    return h, z


def loss(weights, X, y):
    h, _ = forward(weights, X)
    y_hat = np.clip(h[-1][0], 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def gradients(weights, X, y):
    """Backpropagation — the gradient of the cross-entropy w.r.t. every weight matrix."""
    h, _ = forward(weights, X)
    b = X.shape[0]
    # for a sigmoid output and cross-entropy, ∇_{z_K} L comes out as (ŷ − y)/b
    grad_z = (h[-1] - y.reshape(1, -1)) / b
    grad_W = [None] * len(weights)
    for k in range(len(weights) - 1, -1, -1):
        h_with_bias = np.vstack([h[k], np.ones((1, h[k].shape[1]))])
        grad_W[k] = grad_z @ h_with_bias.T
        if k > 0:
            grad_h = (weights[k][:, :-1]).T @ grad_z
            grad_z = grad_h * h[k] * (1 - h[k])          # f'(z) = f(z)(1−f(z))
    return grad_W


def gradient_check(weights, X, y, eps=1e-5):
    """Compares the analytical gradient with a numerical derivative."""
    analytic = gradients(weights, X, y)
    worst = 0.0
    for k, W in enumerate(weights):
        for _ in range(20):
            i = np.random.randint(W.shape[0])
            j = np.random.randint(W.shape[1])
            original = W[i, j]
            W[i, j] = original + eps
            up = loss(weights, X, y)
            W[i, j] = original - eps
            down = loss(weights, X, y)
            W[i, j] = original
            numeric = (up - down) / (2 * eps)
            worst = max(worst, abs(numeric - analytic[k][i, j]))
    return worst


def train(weights, X, y, eta=0.5, batch=32, epochs=400, generator=None):
    generator = generator or np.random.default_rng(0)
    curve = []
    for _ in range(epochs):
        order = generator.permutation(X.shape[0])
        for start in range(0, X.shape[0], batch):
            index = order[start:start + batch]
            for W, grad in zip(weights, gradients(weights, X[index], y[index])):
                W -= eta * grad
        curve.append(loss(weights, X, y))
    return np.array(curve)


def accuracy(weights, X, y):
    h, _ = forward(weights, X)
    return float(np.mean((h[-1][0] > 0.5) == (y > 0.5)))


def main():
    generator = np.random.default_rng(0)
    X, y = make_moons(n_samples=600, noise=0.22, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y.astype(float), test_size=0.3,
                                              random_state=0)

    weights = init_weights([2, 16, 16, 1], generator)
    print(f"largest difference between the analytical and the numerical gradient: "
          f"{gradient_check(weights, X_tr[:32], y_tr[:32]):.2e}")

    curve = train(weights, X_tr, y_tr, generator=generator)
    print(f"training accuracy: {accuracy(weights, X_tr, y_tr):.3f}")
    print(f"test accuracy: {accuracy(weights, X_te, y_te):.3f}")

    grid_x, grid_y = np.meshgrid(np.linspace(-1.6, 2.6, 300),
                                 np.linspace(-1.2, 1.8, 300))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    h, _ = forward(weights, points)
    boundary = h[-1][0].reshape(grid_x.shape)

    _, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(curve, color="#c73e1d")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss",
                title="training loss")
    axes[1].contourf(grid_x, grid_y, boundary, levels=20, cmap="coolwarm",
                     alpha=0.7)
    axes[1].scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm",
                    edgecolor="black", linewidth=0.4, s=18)
    axes[1].set(title="decision boundary (test data)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
