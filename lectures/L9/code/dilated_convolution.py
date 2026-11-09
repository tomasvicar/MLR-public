"""The dilated convolution and the growth of the receptive field.

The slide says that a dilated convolution aggregates context at several
scales and that the receptive field grows without down- and up-sampling. The
script shows this with numbers:

1. a dilated convolution with factor delta = a convolution with a kernel into
   which delta-1 zeros have been inserted between the weights (hence the name
   "à trous") — verified with an `assert` against the direct sum from the
   slide's formula, the cross-correlation of the course:
       z(i, j) = sum_{u, v} x(i + delta*u, j + delta*v) w(u, v),
   u, v = -1, 0, 1 (window centred at (i, j), zeros outside the image), and
   against `scipy.signal.correlate2d`,
2. an F x F kernel with factor delta covers an area of (F-1)*delta + 1,
3. a chain of three 3x3 convolutions with delta = 1, 2, 4 has a receptive field
   of 3 -> 7 -> 15, measured on the impulse response (not computed from the
   formula),
4. for comparison: three ordinary 3x3 convolutions give only 3 -> 5 -> 7.

    MPLBACKEND=Agg uv run python lectures/L9/code/dilated_convolution.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d

SIZE = 81


def dilate(w, delta):
    """Insert delta-1 zeros between the kernel weights."""
    F = w.shape[0]
    large = np.zeros(((F - 1) * delta + 1, (F - 1) * delta + 1))
    large[::delta, ::delta] = w
    return large


def dilated_convolution_by_definition(x, w, delta):
    """Straight from the formula on the slide — slow, but without the kernel trick."""
    r = w.shape[0] // 2                       # offsets u, v = -r … r
    out = np.zeros_like(x)
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            s = 0.0
            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    a, b = i + delta * u, j + delta * v
                    if 0 <= a < n and 0 <= b < m:
                        s += x[a, b] * w[u + r, v + r]
            out[i, j] = s
    return out


def receptive_field(deltas, F=3):
    image = np.zeros((SIZE, SIZE))
    image[SIZE // 2, SIZE // 2] = 1.0
    masks, sizes = [], []
    for delta in deltas:
        image = correlate2d(image, dilate(np.ones((F, F)), delta), mode="same")
        rows = np.nonzero(image.any(axis=1))[0]
        sizes.append(int(rows[-1] - rows[0] + 1))
        masks.append((image > 0).astype(float))
    return masks, sizes


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 12))
    w = rng.normal(size=(3, 3))               # not symmetric: a flip would show
    for delta in (1, 2, 3):
        fast = correlate2d(x, dilate(w, delta), mode="same")
        assert np.allclose(fast, dilated_convolution_by_definition(x, w, delta))
        print(f"delta = {delta}: a 3x3 kernel covers "
              f"{(3 - 1) * delta + 1} x {(3 - 1) * delta + 1} pixels, "
              f"there are still 9 weights  ✓")

    masks, sizes = receptive_field([1, 2, 4])
    assert sizes == [3, 7, 15], sizes
    print(f"chain delta = 1, 2, 4:  receptive field {sizes}")
    _, without_dilation = receptive_field([1, 1, 1])
    print(f"chain without dilation: receptive field {without_dilation}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    r, centre = 8, SIZE // 2
    for ax, mask, v, delta in zip(axes, masks, sizes, (1, 2, 4)):
        ax.imshow(mask[centre - r:centre + r + 1, centre - r:centre + r + 1],
                  cmap="Blues", vmin=0, vmax=1.4)
        ax.set_title(f"after a convolution with delta = {delta}:  {v} x {v}",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
