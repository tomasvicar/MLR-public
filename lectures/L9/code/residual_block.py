"""The residual block and its effect on the gradient — purely in NumPy.

The slides claim two things and this script measures both of them:

1. The gradient through a deep stack: we measure the norm of the
   **back-propagated** gradient ‖∂L/∂h_l‖ with respect to the output h_l of
   layer l, at initialization — the product of Jacobians from the slide. At
   the output the two networks start out similar; in the plain network the
   signal weakens on its way to the input, in the residual one the identity
   carries it and it even grows.
2. "The deeper model should be able to perform at least as well as the
   shallower model, but deeper models are harder to optimize" — three networks
   with the **same** weights at the start (where they share layers), the same
   data and the same optimizer: the shallow plain network learns, the deep
   plain one gets stuck (degradation as described by He et al.), the deep
   residual one learns again.

The only difference between the two deep networks is a single line:

    h = h + tanh(W h + b)      instead of      h = tanh(W h + b)

Deep networks are chaotic in floating point: a difference in the last bit of
one matrix product (different CPUs and BLAS libraries sum in a different
order) changes the final accuracy of the 30-layer plain network by several
percent. So that the script prints exactly the numbers of the slides on every
computer, the matrix product, exp and tanh are written below with exactly
rounded operations in a fixed order (`mm`, `exp`, `tanh`). With `@` and
`np.tanh` the story is the same, only the last digits differ.

    MPLBACKEND=Agg uv run python lectures/L9/code/residual_block.py
"""

import matplotlib.pyplot as plt
import numpy as np

DEPTH = 30
SHALLOW = 3      # shallow plain network for comparison
WIDTH = 16
LN2 = 0.6931471805599453


def exp(x):
    """exp(x) from +, ×, ÷ and ldexp only (identical bits on every machine)."""
    x = np.clip(x, -700.0, 700.0)
    k = np.rint(x / LN2)
    r = x - k * LN2
    p = np.ones_like(r)
    for n in range(14, 0, -1):
        p = 1.0 + p * r / n
    return np.ldexp(p, k.astype(np.int64))


def tanh(x):
    return 1.0 - 2.0 / (exp(2.0 * x) + 1.0)


def mm(A, B):
    """A @ B summed in a fixed order (no BLAS)."""
    return (A[:, :, None] * B[None, :, :]).sum(axis=1)


def data(n=400, noise=0.20, seed=0):
    """Two interleaved half-moons (binary classification)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n // 2)
    X = np.vstack([np.c_[np.cos(t), np.sin(t)],
                   np.c_[1 - np.cos(t), 0.4 - np.sin(t)]])
    X = X + rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
    return np.round((X - X.mean(0)) / X.std(0), 6), y


def initialize(depth=DEPTH, width=WIDTH, seed=0):
    rng = np.random.default_rng(seed)
    par = {"Win": rng.normal(0, np.sqrt(0.5), (2, width)),
           "bin": np.zeros(width),
           "Wout": rng.normal(0, np.sqrt(1 / width), (width, 1)),
           "bout": np.zeros(1)}
    for l in range(depth):
        par[f"W{l}"] = rng.normal(0, np.sqrt(1 / width), (width, width))
        par[f"b{l}"] = np.zeros(width)
    return {k: np.round(v, 8) for k, v in par.items()}


def forward_backward(par, X, y, depth=DEPTH, residual=False):
    """Forward pass, cross-entropy and the gradients of all the weights."""
    h = tanh(mm(X, par["Win"]) + par["bin"])
    hs, acts = [h], []
    for l in range(depth):
        a = tanh(mm(h, par[f"W{l}"]) + par[f"b{l}"])
        h = h + a if residual else a            # <- the only difference
        acts.append(a)
        hs.append(h)
    p = 1.0 / (1.0 + exp(-(mm(h, par["Wout"]) + par["bout"]).ravel()))
    loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    dlogit = ((p - y) / len(y))[:, None]
    grad = {"Wout": mm(hs[-1].T, dlogit), "bout": dlogit.sum(0)}
    dh = mm(dlogit, par["Wout"].T)
    dh_norms = []
    for l in reversed(range(depth)):
        dh_norms.append(float(np.sqrt((dh * dh).sum())))   # ‖∂L/∂h_l‖, l = l+1 here
        dz = dh * (1.0 - acts[l] * acts[l])
        grad[f"W{l}"] = mm(hs[l].T, dz)
        grad[f"b{l}"] = dz.sum(0)
        # in the residual network the identity adds the gradient itself — the
        # layers would have to cancel it exactly to stop it
        dh = mm(dz, par[f"W{l}"].T) + (dh if residual else 0.0)
    dzin = dh * (1.0 - hs[0] * hs[0])
    grad["Win"] = mm(X.T, dzin)
    grad["bin"] = dzin.sum(0)
    return loss, grad, p, np.array(dh_norms[::-1])   # index 0 = layer 1


def train(X, y, residual, depth=DEPTH, epochs=500, lr=0.004, seed=0):
    par = initialize(depth=depth, seed=seed)
    m = {k: np.zeros_like(v) for k, v in par.items()}
    v = {k: np.zeros_like(val) for k, val in par.items()}
    curve = []
    beta1_t = beta2_t = 1.0
    for t in range(1, epochs + 1):                      # Adam
        loss, grad, _, _ = forward_backward(par, X, y, depth=depth,
                                            residual=residual)
        curve.append(loss)
        beta1_t *= 0.9
        beta2_t *= 0.999
        for k in par:
            g = grad[k].reshape(par[k].shape)
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.999 * v[k] + 0.001 * (g * g)
            par[k] = par[k] - lr * (m[k] / (1 - beta1_t)) / (
                np.sqrt(v[k] / (1 - beta2_t)) + 1e-8)
    loss, _, p, _ = forward_backward(par, X, y, depth=depth, residual=residual)
    return np.array(curve), loss, float(np.mean((p > 0.5) == (y > 0.5)))


def main():
    X, y = data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # shallow plain network: the same first matrices, just SHALLOW layers
    curve, loss, accuracy = train(X, y, residual=False, depth=SHALLOW)
    ax2.plot(curve, color="#8895a0", linewidth=2, linestyle="--",
             label=f"plain, {SHALLOW} layers (acc {accuracy:.1%})")
    print(f"shallow   ({SHALLOW} layers) after 500 epochs: loss {loss:.4f}, "
          f"accuracy {accuracy:.2%}")

    for residual, colour, name in ((False, "#c73e1d", "plain"),
                                   (True, "#007f86", "residual")):
        par = initialize()
        _, _, _, norms = forward_backward(par, X, y, residual=residual)
        ax1.semilogy(range(1, DEPTH + 1), norms, "o-", color=colour,
                     markersize=4, label=name)
        print(f"{name:9s} ‖dL/dh_l‖: l = {DEPTH} (output) {norms[-1]:.4f}, "
              f"l = 1 {norms[0]:.4f}, ratio {norms[0] / norms[-1]:.3g}x")

        curve, loss, accuracy = train(X, y, residual)
        ax2.plot(curve, color=colour, linewidth=2,
                 label=f"{name}, {DEPTH} layers (acc {accuracy:.1%})")
        print(f"{name:9s} after 500 epochs: loss {loss:.4f}, "
              f"accuracy {accuracy:.2%}")

    ax1.set_xlabel("layer l (1 = at the input, 30 = output)")
    ax1.set_ylabel(r"$\|\partial \mathcal{L}/\partial \mathbf{h}_l\|$ at initialization")
    ax1.set_title(f"{DEPTH} layers, tanh")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training loss")
    ax2.set_title("same data, same optimizer")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
