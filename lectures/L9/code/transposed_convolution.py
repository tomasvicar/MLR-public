"""Convolution as a sparse matrix product, and the transposed convolution.

Exactly the procedure from the slides "Convolution matrix C" and "Transposed
convolution", only with concrete numbers: N = 3, F = 3, so M = N + (F-1) = 5
and C is of size M^2 x M^2 = 25 x 25.

The convolution is the cross-correlation of the course (no kernel flip):
    z(i, j) = sum_{u, v} x(i + u, j + v) w(u, v),
here with the window anchored at its top-left corner, u, v = 0 … F-1.

1. Build the convolution matrix C from the kernel W (each row = one kernel
   position).
2. Verify with an `assert` that C @ vec(X) gives the same as
   `scipy.signal.correlate2d`.
3. For stride S = 2, drop the corresponding **rows** (not columns!) — C is
   then of size ceil(M/2)^2 x M^2 = 9 x 25, is not square and therefore has no
   inverse (for any kernel).
4. The transposed convolution vec(Y) = C^T vec(Z) takes the 3 x 3 map back to
   a 5 x 5 map. The shape is restored, the values are not — which is why this
   is not a "deconvolution". In a network the kernel inside C^T is learned;
   here it is initialized with the convolution's own kernel.
5. The same C^T vec(Z) computed as a **scatter**: every entry of Z multiplies
   the kernel and adds it into Y at its own position (stride 2). The copies
   overlap unevenly (1, 2 or 4 per pixel) — hence the checkerboard artefact.

    MPLBACKEND=Agg uv run python lectures/L9/code/transposed_convolution.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d

N, F = 3, 3
M = N + (F - 1)


def convolution_matrix(w, M, stride=1):
    """A row of the matrix = one kernel position: z(i,j) = sum_uv w(u,v) x(i+u, j+v)."""
    rows = []
    for i in range(0, M, stride):
        for j in range(0, M, stride):
            r = np.zeros((M, M))
            for u in range(w.shape[0]):
                for v in range(w.shape[1]):
                    if i + u < M and j + v < M:
                        r[i + u, j + v] = w[u, v]
            rows.append(r.ravel())
    return np.array(rows)


def main():
    rng = np.random.default_rng(3)
    W = np.round(rng.uniform(-1, 1, (F, F)), 2)
    image = np.round(rng.uniform(0, 1, (N, N)), 2)
    X = np.zeros((M, M))                      # zero padding: N x N -> M x M
    X[1:1 + N, 1:1 + N] = image

    C = convolution_matrix(W, M)
    Z = (C @ X.ravel()).reshape(M, M)
    print(f"C is of size {C.shape}, non-zero entries {int((C != 0).sum())} "
          f"out of {C.size} ({(C != 0).mean():.1%})")

    # check against scipy: the cross-correlation, cropped to M x M
    ref = correlate2d(X, W, mode="full")[F - 1:F - 1 + M, F - 1:F - 1 + M]
    assert np.allclose(Z, ref)
    print("C @ vec(X) == scipy.signal.correlate2d  ✓")
    assert np.allclose(Z[:N, :N], correlate2d(image, W, mode="same"))
    print(f"the top left {N}x{N} block of Z is the 'same' output of the "
          f"unpadded image:\n{np.round(Z[:N, :N], 3)}")

    # stride 1: here C is square, even upper triangular with w11 on the
    # diagonal, so it is invertible because w11 != 0 — only badly conditioned.
    # A kernel with w11 = 0 (e.g. the Laplacian) makes it singular already.
    print(f"det(C) = {np.linalg.det(C):.3e} = w11^(M^2) = "
          f"{W[0, 0] ** (M * M):.3e},  cond(C) = {np.linalg.cond(C):.1f}")
    laplace = np.array([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])
    print(f"the Laplacian kernel gives C of rank "
          f"{np.linalg.matrix_rank(convolution_matrix(laplace, M))} of {M * M}")

    # stride 2: rows drop out and the matrix stops being square. On the padded
    # X the last row and column of kernel positions lie entirely in the zero
    # padding, so a 5 x 5 map without a zero border is used here.
    X2 = np.round(np.random.default_rng(11).uniform(0.1, 1, (M, M)), 2)
    C2 = convolution_matrix(W, M, stride=2)
    Z2 = (C2 @ X2.ravel()).reshape(3, 3)
    assert C2.shape == (9, M * M)
    assert np.allclose(Z2, (C @ X2.ravel()).reshape(M, M)[::2, ::2])
    print(f"stride 2: C is of size {C2.shape} -> not square, "
          f"C^-1 does not exist; Z =\n{np.round(Z2, 2)}")

    # the transposed convolution: C^T of the same sparse shape (learned kernel)
    Y = (C2.T @ Z2.ravel()).reshape(M, M)
    assert Y.shape == X2.shape
    print(f"C^T @ vec(Z): {Z2.shape} -> {Y.shape}  ✓ (the shape fits, the values do not)")
    print(f"relative deviation from X: "
          f"{np.linalg.norm(Y - X2) / np.linalg.norm(X2):.2f}")

    # the same as a scatter: Z[a, b] * the kernel placed at (2a, 2b), cropped at the edge
    Y_scatter = np.zeros((M, M))
    copies = np.zeros((M, M))
    for a in range(3):
        for b in range(3):
            i, j = 2 * a, 2 * b
            di, dj = min(F, M - i), min(F, M - j)
            Y_scatter[i:i + di, j:j + dj] += Z2[a, b] * W[:di, :dj]
            copies[i:i + di, j:j + dj] += 1
    assert np.allclose(Y_scatter, Y)
    print("C^T vec(Z) == the sum of kernel copies weighted by Z  ✓; copies per pixel:",
          {int(v): int((copies == v).sum()) for v in np.unique(copies)})

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, matrix, title in ((axes[0], X, f"X after padding {M}x{M}"),
                              (axes[1], Z, f"Z = C vec(X), {M}x{M}"),
                              (axes[2], X2, f"X without padding {M}x{M}"),
                              (axes[3], Z2, "Z with stride 2, 3x3"),
                              (axes[4], Y, f"Y = C^T vec(Z), {M}x{M}")):
        m = np.abs(matrix).max()
        ax.imshow(matrix, cmap="RdBu_r", vmin=-m, vmax=m)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
