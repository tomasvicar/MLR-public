"""Entropy, joint entropy and mutual information from histograms.

Port of the original Colab notebook `pomocne_colaby/entropy.ipynb`.

    uv run python lectures/L3/code/entropy_example.py
"""

import numpy as np

BINS = 30


def entropy(x: np.ndarray) -> float:
    """Entropy [bits] of a 1D sample, distribution approximated by a histogram."""
    hist, _ = np.histogram(x, bins=BINS)
    p = hist / hist.sum() + np.finfo(float).eps   # so that log(0) does not blow up
    return float(-np.sum(p * np.log2(p)))


def joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Joint entropy [bits] from a 2D histogram of the pairs (x_i, y_i)."""
    hist, _, _ = np.histogram2d(x, y, bins=BINS)
    p = hist / hist.sum() + np.finfo(float).eps
    return float(-np.sum(p * np.log2(p)))


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    return entropy(x) + entropy(y) - joint_entropy(x, y)


def main() -> None:
    rng = np.random.default_rng(0)

    # --- 1D: peaked (low entropy) vs. uniform (high entropy) ---
    peaked = rng.normal(loc=0.0, scale=0.3, size=1000)
    uniform = rng.uniform(low=-3.0, high=3.0, size=1000)
    print(f"entropy, peaked  distribution: {entropy(peaked):7.4f}")
    print(f"entropy, uniform distribution: {entropy(uniform):7.4f}")

    # --- 2D: dependent (low joint entropy) vs. independent (high joint entropy) ---
    x_dep = rng.normal(size=2000)
    y_dep = 2 * x_dep + rng.normal(scale=0.2, size=2000)
    x_ind = rng.normal(size=2000)
    y_ind = rng.normal(size=2000)

    print(f"joint entropy, dependent   pair: {joint_entropy(x_dep, y_dep):7.4f}")
    print(f"joint entropy, independent pair: {joint_entropy(x_ind, y_ind):7.4f}")
    print(f"mutual information, dependent   pair: {mutual_information(x_dep, y_dep):7.4f}")
    # The independent pair does not give exactly 0: a histogram of 2000 samples in
    # 30x30 bins overestimates MI (binning bias). A kNN estimator gives ~0 here.
    print(f"mutual information, independent pair: {mutual_information(x_ind, y_ind):7.4f}")


if __name__ == "__main__":
    main()
