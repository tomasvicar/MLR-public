"""Dropout and its effect on overfitting — all in numpy.

A needlessly large network (two hidden layers of 128 ReLU neurons) is trained on
120 noisy samples and validated on another 780. Both variants memorize the
training data (training accuracy 1.0), but the one with dropout generalizes a few
percentage points better.

Dropout is implemented exactly as in the lecture:

* the state of every neuron is a Bernoulli variable with probability `p` of being
  kept (1 = keep, 0 = disable),
* the output of the layer is divided by `p` (inverted dropout — the expected value is conserved), so that
  nothing has to be adjusted at prediction time,
* the output layer is never switched off,
* in the backward pass the gradient is multiplied by the same mask.

    MPLBACKEND=Agg uv run python lectures/L7/code/dropout_overfitting.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


class Net:
    """An MLP with ReLU hidden layers, a sigmoid output and dropout."""

    def __init__(self, sizes, p_hidden=1.0, seed=0):
        self.generator = np.random.default_rng(seed)
        self.weights = [self.generator.normal(scale=np.sqrt(2.0 / n_in),
                                              size=(n_out, n_in + 1))
                        for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.p_hidden = p_hidden

    def forward(self, X, training):
        h = [X.T]
        masks = []
        for k, W in enumerate(self.weights):
            h_with_bias = np.vstack([h[-1], np.ones((1, h[-1].shape[1]))])
            a = W @ h_with_bias
            last = k == len(self.weights) - 1
            hk = sigmoid(a) if last else np.maximum(0.0, a)
            if training and not last and self.p_hidden < 1.0:
                # 1 = keep, 0 = disable; dividing by p preserves the mean
                mask = (self.generator.random(hk.shape) < self.p_hidden)
                hk = hk * mask / self.p_hidden
                masks.append(mask / self.p_hidden)
            else:
                masks.append(None)
            h.append(hk)
        return h, masks

    def gradients(self, X, y):
        h, masks = self.forward(X, training=True)
        grad_a = (h[-1] - y.reshape(1, -1)) / X.shape[0]
        grad_W = [None] * len(self.weights)
        for k in range(len(self.weights) - 1, -1, -1):
            h_with_bias = np.vstack([h[k], np.ones((1, h[k].shape[1]))])
            grad_W[k] = grad_a @ h_with_bias.T
            if k > 0:
                grad_h = self.weights[k][:, :-1].T @ grad_a
                if masks[k - 1] is not None:
                    grad_h = grad_h * masks[k - 1]   # only the active neurons
                grad_a = grad_h * (h[k] > 0)     # f'(a) for ReLU
        return grad_W

    def accuracy(self, X, y):
        h, _ = self.forward(X, training=False)
        return float(np.mean((h[-1][0] > 0.5) == (y > 0.5)))


def train(net, X_tr, y_tr, X_val, y_val, epochs=200, eta=0.02, batch=16):
    history = []
    for _ in range(epochs):
        order = net.generator.permutation(X_tr.shape[0])
        for start in range(0, X_tr.shape[0], batch):
            index = order[start:start + batch]
            for W, grad in zip(net.weights, net.gradients(X_tr[index], y_tr[index])):
                W -= eta * grad
        history.append((net.accuracy(X_tr, y_tr), net.accuracy(X_val, y_val)))
    return np.array(history)


def main():
    X, y = make_classification(n_samples=900, n_features=40, n_informative=8,
                               n_redundant=6, flip_y=0.05, class_sep=1.0,
                               random_state=0)
    X = StandardScaler().fit_transform(X)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y.astype(float),
                                                train_size=120, random_state=0)

    _, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, p in zip(axes, (1.0, 0.5)):
        net = Net([X.shape[1], 128, 128, 1], p_hidden=p, seed=0)
        history = train(net, X_tr, y_tr, X_val, y_val)
        name = "no dropout" if p == 1.0 else f"dropout, p = {p}"
        print(f"{name:18s}  train = {history[-1, 0]:.3f}   "
              f"validation = {history[-1, 1]:.3f}   "
              f"gap = {history[-1, 0] - history[-1, 1]:.3f}")
        ax.plot(history[:, 0], color="#c73e1d", label="training accuracy")
        ax.plot(history[:, 1], color="#3a8a52", label="validation accuracy")
        ax.set(title=name, xlabel="epoch", ylim=(0.5, 1.02))
        ax.legend(loc="lower right")
    axes[0].set_ylabel("accuracy")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
