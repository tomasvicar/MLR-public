"""A convolutional layer written from scratch in NumPy — and what it learns.

Code example for the eighth lecture (Convolutional networks 1). The network is
deliberately minimal, so that its weights show directly what the convolutional
layer has learned:

    32x32x3 -> conv 5x5, K=12 -> ReLU -> global max pool -> FC 8 -> softmax

The data are synthetic: 3-5 coloured strokes in noise, the class is given by
their orientation (eight orientations 22.5 degrees apart). The task is purely
about orientation and it is noisy, so it pays for the layer to learn matched
filters — oriented colour edges. An L2 penalty on the weights ("weight
decay"; it is added to the gradient before Adam, as
`torch.optim.Adam(weight_decay=...)` does — not the decoupled decay of
AdamW) meanwhile suppresses the random
component the filters started with; without it (or with a small one) most of
the filters stay coloured noise, even though the network classifies just as
well.

The convolution is computed through `im2col`: all the windows are pulled out
as a view into memory (`as_strided`) into a single matrix, and the convolution
is then a single matrix product. The backward pass is written by hand, the
optimizer is Adam.

The same code is used by the figure generator of the lecture
(`make_figures.py`), so the filters and activation maps on the slides are
really learned, not drawn.

    MPLBACKEND=Agg uv run python lectures/L8/code/cnn_filters.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

N_CLASSES = 8                                     # eight stroke orientations
ANGLES = tuple(180.0 / N_CLASSES * i for i in range(N_CLASSES))
SIZE = 32                                         # W_x = H_x
F, D, K = 5, 3, 12                                # filter 5x5x3, K filters
AMPLITUDE = 0.8                                   # stroke contrast against the background
NOISE = 0.12                                      # standard deviation of the noise


# ---------------------------------------------------------------- data ----

def _stroke(img, p0, p1, colour, thickness):
    """Draw a line segment with soft edges into the image."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.stack([xx, yy], axis=-1).astype(np.float64)
    direction = p1 - p0
    t = np.clip(((points - p0) @ direction) / float(direction @ direction), 0.0, 1.0)
    distance = np.linalg.norm(points - (p0 + t[..., None] * direction), axis=-1)
    coverage = AMPLITUDE * np.clip(1.0 - (distance - thickness / 2 + 0.5), 0, 1)
    img *= 1.0 - coverage[..., None]
    img += coverage[..., None] * colour


def make_image(cls, rng):
    """One image of the given class: 3-5 coloured strokes of one orientation in noise."""
    img = np.full((SIZE, SIZE, D), rng.uniform(0.4, 0.6))
    for _ in range(int(rng.integers(3, 6))):
        angle = np.deg2rad(ANGLES[cls] + rng.normal(0.0, 5.0))
        direction = np.array([np.cos(angle), np.sin(angle)])
        centre = rng.uniform(6, SIZE - 6, 2)
        length = rng.uniform(10, 17)
        colour = hsv_to_rgb([rng.uniform(), rng.uniform(0.6, 1.0),
                             rng.uniform(0.75, 1.0)])
        _stroke(img, centre - direction * length / 2, centre + direction * length / 2,
                colour, rng.uniform(1.6, 2.6))
    return np.clip(img + rng.normal(0.0, NOISE, img.shape), 0.0, 1.0)


def make_data(n, rng):
    y = rng.integers(0, N_CLASSES, n)
    x = np.stack([make_image(t, rng) for t in y])
    return x.astype(np.float32), y


# ------------------------------------------------------- convolution ----

def im2col(x, f):
    """(N,H,W,D) -> (N,Ho,Wo,f*f*D); all convolution windows with stride 1."""
    x = np.ascontiguousarray(x)
    n, h, w, d = x.shape
    ho, wo = h - f + 1, w - f + 1
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(n, ho, wo, f, f, d),
        strides=(s[0], s[1], s[2], s[1], s[2], s[3]))
    return windows.reshape(n, ho, wo, f * f * d)


def forward(m, x):
    """Convolution, ReLU and global max pooling; also returns the intermediates."""
    c = {"x": x, "cols": im2col(x, F)}
    c["a"] = np.maximum(c["cols"] @ m["W"] + m["b"], 0.0)        # N x 28 x 28 x K
    n, ho, wo, k = c["a"].shape
    flat = c["a"].reshape(n, ho * wo, k)
    c["where"] = flat.argmax(axis=1)          # where in the map the strongest response was
    c["g"] = flat.max(axis=1)
    c["p"] = _softmax(c["g"] @ m["Wf"] + m["bf"])
    return c


def backward(m, c, y):
    """Gradient of the cross-entropy w.r.t. the filter and output layer weights."""
    n = len(y)
    dz = c["p"].copy()
    dz[np.arange(n), y] -= 1.0
    dz /= n
    g = {"Wf": c["g"].T @ dz, "bf": dz.sum(axis=0)}

    da = np.zeros((n, c["a"].shape[1] * c["a"].shape[2], K), np.float32)
    np.put_along_axis(da, c["where"][:, None, :], (dz @ m["Wf"].T)[:, None, :], 1)
    da = da.reshape(c["a"].shape) * (c["a"] > 0)          # back through the ReLU
    g["W"] = c["cols"].reshape(-1, F * F * D).T @ da.reshape(-1, K)
    g["b"] = da.sum(axis=(0, 1, 2))
    return g


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ training ----

def train(seed=0, n_train=2000, epochs=120, batch=50, mu=2e-3, decay=2e-2,
          quiet=True, frames=None):
    """Adam with an L2 penalty `decay` on the weights. Returns the model, the test data and the accuracy history.

    When `frames` is a list, a copy of the filter weights `W` is appended to it
    after every epoch (and once before the training starts) — the figure
    generator turns them into an animation of how the filters emerge out of
    random noise.
    """
    rng = np.random.default_rng(seed)
    x, y = make_data(n_train, rng)
    xt, yt = make_data(400, rng)
    m = {"W": rng.normal(0, np.sqrt(2 / (F * F * D)), (F * F * D, K)),
         "b": np.full(K, 0.1),   # a small positive bias against dying ReLUs
         "Wf": rng.normal(0, np.sqrt(2 / K), (K, N_CLASSES)),
         "bf": np.zeros(N_CLASSES)}
    m = {k: v.astype(np.float32) for k, v in m.items()}
    first = {k: np.zeros_like(v) for k, v in m.items()}
    second = {k: np.zeros_like(v) for k, v in m.items()}
    history, step = [], 0
    if frames is not None:
        frames.append(m["W"].copy())
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y) - batch + 1, batch):
            j = order[i:i + batch]
            g = backward(m, forward(m, x[j]), y[j])
            step += 1
            for k in g:
                grad = g[k] + (decay * m[k] if k.startswith("W") else 0.0)
                first[k] = 0.9 * first[k] + 0.1 * grad
                second[k] = 0.999 * second[k] + 0.001 * grad ** 2
                m[k] -= mu * (first[k] / (1 - 0.9 ** step)) / (
                    np.sqrt(second[k] / (1 - 0.999 ** step)) + 1e-8)
        history.append(float((forward(m, xt)["p"].argmax(axis=1) == yt).mean()))
        if frames is not None:
            frames.append(m["W"].copy())
        if not quiet:
            print(f"epoch {len(history):2d}  test accuracy {history[-1]:.3f}")
    return m, (xt, yt), history


def filter_order(m, x):
    """Filters sorted from the strongest by their mean activation on the data."""
    return np.argsort(-forward(m, x)["a"].mean(axis=(0, 1, 2)))


def filters_from_weights(w):
    """Weight matrix (F*F*D, K) as K colour images 5x5, each stretched to [0,1]."""
    f = w.reshape(F, F, D, K).transpose(3, 0, 1, 2)
    low = f.min(axis=(1, 2, 3), keepdims=True)
    high = f.max(axis=(1, 2, 3), keepdims=True)
    return (f - low) / (high - low)


def filters(m):
    """The learned filters of the model as K colour images 5x5."""
    return filters_from_weights(m["W"])


if __name__ == "__main__":
    model, (x_test, y_test), history = train(quiet=False)
    print(f"final test accuracy: {history[-1]:.3f}")

    order = filter_order(model, x_test[:200])
    filter_images = filters(model)[order]
    maps = forward(model, x_test[:1])["a"][0][..., order]

    fig, axes = plt.subplots(2, K, figsize=(11, 2.6))
    for i in range(K):
        axes[0, i].imshow(filter_images[i], interpolation="nearest")
        axes[1, i].imshow(maps[..., i], cmap="gray")
        axes[0, i].set_axis_off()
        axes[1, i].set_axis_off()
    axes[0, 0].set_title("learned 5x5x3 filters", loc="left", fontsize=9)
    axes[1, 0].set_title("activation maps of one input", loc="left", fontsize=9)

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.show()
