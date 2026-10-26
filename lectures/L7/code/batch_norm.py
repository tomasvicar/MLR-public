"""Batch normalization from scratch in numpy.

Implements exactly the steps 1.–7. from the slides:

1. the mean and the variance of the mini-batch for every feature separately, plus
   the running statistics of the whole dataset (exponential forgetting with the
   parameter rho),
2. normalization of the mini-batch,
3. a transformation by the learnable parameters gamma (scale) and beta (shift),
4. the output goes into the next layer,
5. gamma and beta are learned by the chain rule,
6. it repeats throughout training,
7. after training, the running statistics fold into a single linear transformation
   for the prediction phase — and the script verifies that it gives the same result
   as normalizing by the running statistics.

It shows two things from the lecture: BN allows a larger learning rate and speeds
up training, and at prediction time it costs nothing extra.

    MPLBACKEND=Agg uv run python lectures/L7/code/batch_norm.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPS = 1e-5
RHO = 0.9            # exponential forgetting of the running statistics


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


class Net:
    """An MLP with ReLU layers, optionally with batch normalization."""

    def __init__(self, sizes, batch_norm=False, seed=0):
        generator = np.random.default_rng(seed)
        self.generator = generator
        self.weights = [generator.normal(scale=np.sqrt(2.0 / n_in),
                                         size=(n_out, n_in + 1))
                        for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.batch_norm = batch_norm
        hidden = sizes[1:-1]
        self.gamma = [np.ones((n, 1)) for n in hidden]
        self.beta = [np.zeros((n, 1)) for n in hidden]
        self.mu_D = [np.zeros((n, 1)) for n in hidden]
        self.var_D = [np.ones((n, 1)) for n in hidden]

    def forward(self, X, training):
        h = [X.T]
        cache = []
        for k, W in enumerate(self.weights):
            a = W @ np.vstack([h[-1], np.ones((1, h[-1].shape[1]))])
            last = k == len(self.weights) - 1
            if last:
                h.append(sigmoid(a))
                cache.append(None)
                continue
            if self.batch_norm:
                if training:
                    mu_B = a.mean(axis=1, keepdims=True)
                    var_B = ((a - mu_B) ** 2).mean(axis=1, keepdims=True)
                    self.mu_D[k] = RHO * self.mu_D[k] + (1 - RHO) * mu_B
                    self.var_D[k] = RHO * self.var_D[k] + (1 - RHO) * var_B
                else:
                    mu_B, var_B = self.mu_D[k], self.var_D[k]
                a_norm = (a - mu_B) / np.sqrt(var_B + EPS)
                a = self.gamma[k] * a_norm + self.beta[k]
                cache.append((a_norm, var_B))
            else:
                cache.append(None)
            h.append(np.maximum(0.0, a))
        return h, cache

    def step(self, X, y, eta):
        h, cache = self.forward(X, training=True)
        m = X.shape[0]
        grad_a = (h[-1] - y.reshape(1, -1)) / m
        for k in range(len(self.weights) - 1, -1, -1):
            grad_W = grad_a @ np.vstack([h[k], np.ones((1, h[k].shape[1]))]).T
            if k > 0:
                grad_h = self.weights[k][:, :-1].T @ grad_a
                grad_a = grad_h * (h[k] > 0)
                if self.batch_norm:
                    a_norm, var_B = cache[k - 1]
                    grad_gamma = (grad_a * a_norm).sum(axis=1, keepdims=True)
                    grad_beta = grad_a.sum(axis=1, keepdims=True)
                    # gradient through the mini-batch normalization (with the forward-pass gamma)
                    g = grad_a * self.gamma[k - 1]
                    grad_a = (g - g.mean(axis=1, keepdims=True)
                              - a_norm * (g * a_norm).mean(axis=1, keepdims=True)
                              ) / np.sqrt(var_B + EPS)
                    self.gamma[k - 1] -= eta * grad_gamma
                    self.beta[k - 1] -= eta * grad_beta
            self.weights[k] -= eta * grad_W

    def accuracy(self, X, y):
        h, _ = self.forward(X, training=False)
        return float(np.mean((h[-1][0] > 0.5) == (y > 0.5)))


def train(net, X_tr, y_tr, X_val, y_val, epochs=60, eta=0.05, batch=32):
    history = []
    for _ in range(epochs):
        order = net.generator.permutation(X_tr.shape[0])
        for start in range(0, X_tr.shape[0], batch):
            index = order[start:start + batch]
            net.step(X_tr[index], y_tr[index], eta)
        history.append(net.accuracy(X_val, y_val))
    return np.array(history)


def check_inference_transform(net, k=0):
    """Step 7: gamma_inf and beta_inf have to give the same value as steps 2.–3."""
    gamma_inf = net.gamma[k] / np.sqrt(net.var_D[k] + EPS)
    beta_inf = net.beta[k] - net.gamma[k] * net.mu_D[k] / np.sqrt(
        net.var_D[k] + EPS)
    a = net.generator.normal(size=(net.gamma[k].size, 5))
    step_by_step = net.gamma[k] * (a - net.mu_D[k]) / np.sqrt(
        net.var_D[k] + EPS) + net.beta[k]
    single_linear = gamma_inf * a + beta_inf
    return float(np.abs(step_by_step - single_linear).max())


def main():
    X, y = make_classification(n_samples=2000, n_features=30, n_informative=10,
                               n_redundant=5, class_sep=0.9, random_state=0)
    X = StandardScaler().fit_transform(X) * 4.0        # badly scaled input
    X_tr, X_val, y_tr, y_val = train_test_split(X, y.astype(float),
                                                test_size=0.3, random_state=0)

    _, ax = plt.subplots(figsize=(7.5, 4.5))
    for batch_norm, color in ((False, "#c73e1d"), (True, "#007f86")):
        net = Net([X.shape[1], 64, 64, 1], batch_norm=batch_norm, seed=0)
        history = train(net, X_tr, y_tr, X_val, y_val)
        label = "with batch norm" if batch_norm else "without batch norm"
        print(f"{label:20s} validation accuracy after 60 epochs = "
              f"{history[-1]:.3f}")
        if batch_norm:
            print("   error of the single linear transformation at prediction time: "
                  f"{check_inference_transform(net):.2e}")
        ax.plot(history, color=color, linewidth=2, label=label)
    ax.set(xlabel="epoch", ylabel="validation accuracy",
           title="Batch normalization on badly scaled inputs")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
