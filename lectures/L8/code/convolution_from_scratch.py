"""Convolution and pooling written from scratch in NumPy.

Code example for the eighth lecture (Convolutional networks 1). It shows three things:

1. the convolution of a 3D filter over a 3D input with stride and zero padding,
   exactly following the formulas from the slides, checked against
   `scipy.signal` (including the sum over the depth D = 3),
2. the difference between convolution (flipped filter) and correlation
   (unflipped) — what the deep learning libraries call "convolution" is the
   correlation,
3. max pooling and average pooling, including the example from the slide.

    MPLBACKEND=Agg uv run python lectures/L8/code/convolution_from_scratch.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, correlate2d
from sklearn.datasets import load_sample_image


def convolve(x, w, stride=1, padding=0, flip=False):
    """Input x (H,W,D), filter w (F,F,D) -> activation map (Ho,Wo).

    The default `flip=False` computes the cross-correlation — the formula on
    the slides and what the libraries compute under the name convolution;
    `flip=True` flips the filter first, which is the true convolution of
    mathematics (mentioned in the notes of the slide).
    """
    f = w.shape[0]
    if flip:
        w = w[::-1, ::-1, :]
    x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)))
    h_out = (x.shape[0] - f) // stride + 1
    w_out = (x.shape[1] - f) // stride + 1
    a = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = x[i * stride:i * stride + f, j * stride:j * stride + f, :]
            a[i, j] = np.sum(patch * w)          # dot product = z, as in a neuron
    return a


def pooling(x, f=2, stride=2, kind="max"):
    """Pooling of a 2D map; output size (W - F) / S + 1."""
    h_out = (x.shape[0] - f) // stride + 1
    w_out = (x.shape[1] - f) // stride + 1
    out = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            window = x[i * stride:i * stride + f, j * stride:j * stride + f]
            out[i, j] = window.max() if kind == "max" else window.mean()
    return out


# --------------------------------------------------- check against scipy ----

rng = np.random.default_rng(0)
x = rng.normal(size=(9, 9, 1))
w = rng.normal(size=(3, 3, 1))

scipy_conv = convolve2d(x[..., 0], w[..., 0], mode="valid")
scipy_corr = correlate2d(x[..., 0], w[..., 0], mode="valid")
assert np.allclose(convolve(x, w), scipy_corr)
assert np.allclose(convolve(x, w, flip=True), scipy_conv)

# A filter with depth: the dot product also sums over the D channels, so a
# 3-channel input gives ONE map (the sum of the three 2D correlations).
x3 = rng.normal(size=(9, 9, 3))
w3 = rng.normal(size=(3, 3, 3))
assert convolve(x3, w3).shape == (7, 7)
assert np.allclose(convolve(x3, w3), sum(correlate2d(x3[..., d], w3[..., d], mode="valid")
                                         for d in range(3)))
assert np.allclose(convolve(x3, w3, flip=True),
                   sum(convolve2d(x3[..., d], w3[..., d], mode="valid") for d in range(3)))
print("correlation and convolution agree with scipy, also over the depth D = 3")

# Output size according to the formula from the slide.
assert convolve(x, w, stride=2).shape == ((9 - 3) // 2 + 1,) * 2
assert convolve(x, w, padding=1).shape == (9, 9)       # (9-3+2)/1+1 = 9
print("output sizes match the formula (W - F + 2P) / S + 1")

# The pooling example from the slide.
example = np.array([[1, 0, 2, 2],
                    [6, 4, 3, 4],
                    [1, 9, 4, 0],
                    [7, 8, 0, 1]], float)
assert np.array_equal(pooling(example, kind="max"), [[6, 4], [9, 4]])
assert np.allclose(pooling(example, kind="avg"), [[2.75, 2.75], [6.25, 1.25]])
print("max pool  ->", pooling(example, kind="max").tolist())
print("avg pool  ->", pooling(example, kind="avg").tolist(), "(rounded to 2.8 / 2.8 / 6.3 / 1.3 on the slide)")


# ------------------------------------------------------ demo on an image ----

image = load_sample_image("china.jpg")[80:400:2, 120:520:2] / 255.0
gray = image.mean(axis=2, keepdims=True)

FILTERS = {
    "vertical edge": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float),
    "horizontal edge": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float),
    "blur": np.ones((3, 3)) / 9,
}

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(image)
axes[0].set_title("input")
for ax, (name, kernel) in zip(axes[1:], FILTERS.items()):
    fmap = convolve(gray, kernel[:, :, None])
    ax.imshow(fmap, cmap="gray")
    ax.set_title(f"{name}\n{fmap.shape[1]}x{fmap.shape[0]}")
for ax in axes:
    ax.set_axis_off()
fig.tight_layout()

# Stride and pooling shrink the map, the depth stays.
fmap = convolve(gray, FILTERS["vertical edge"][:, :, None])
print("map", fmap.shape, "-> max pool 2x2", pooling(fmap).shape)

plt.show()
