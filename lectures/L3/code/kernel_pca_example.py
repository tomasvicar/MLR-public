"""PCA in the primal form, in the dual form, and kernel PCA from scratch.

Port of the original Colab notebook `pomocne_colaby/kernel_pca.ipynb`. The
three variants are computed on the same data — the primal and the dual PCA
give the same components, kernel PCA with an RBF kernel unfolds the two rings.

    uv run python lectures/L3/code/kernel_pca_example.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA

GAMMA = 0.001


def data(n=200, seed=5):
    """Two concentric rings, stretched by a rotation + scaling."""
    rng = np.random.RandomState(seed)
    angle = rng.rand(n) * 2 * np.pi
    length = rng.rand(n) * 40 + 30
    blue = np.stack((np.sin(angle) * length, np.cos(angle) * length)).T

    angle = rng.rand(n) * 2 * np.pi
    length = rng.rand(n) * 40
    red = np.stack((np.sin(angle) * length, np.cos(angle) * length)).T

    rot = np.array([[2 * np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    X = np.concatenate((red, blue), axis=0).astype(float) @ rot.T
    y = np.concatenate((-np.ones(n), np.ones(n))) * -1
    return X, y


def pca_primal(X: np.ndarray) -> np.ndarray:
    """C = (1/N) X^T X, eigenvectors of C are the component directions."""
    X = X - X.mean(axis=0)                # centering is a part of PCA:
    N = X.shape[0]                        # without it C mixes the mean in
    C = (1 / N) * X.T @ X
    lam, W = np.linalg.eigh(C)
    order = np.argsort(lam.real)[::-1]
    return X @ W[:, order].real


def pca_dual(X: np.ndarray) -> np.ndarray:
    """C' = X X^T (Gram matrix), the scores follow from its eigenvectors."""
    X = X - X.mean(axis=0)                # same centering as in the primal form
    C = X @ X.T
    lam, a = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, a = lam[order], a[:, order]
    positive = lam > max(lam[0], 1.0) * 1e-12
    return C @ (a[:, positive] / np.sqrt(lam[positive]))



def kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """RBF kernel k(x, y) = exp(-gamma ||x - y||^2)."""
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


def fit_kernel_pca(X: np.ndarray, gamma: float = GAMMA, n_components: int = 2):
    """Store training centring statistics and normalized positive eigenvectors."""
    K = kernel_matrix(X, X, gamma)
    column_mean = K.mean(axis=0)
    grand_mean = K.mean()
    centered = K - column_mean[None, :] - column_mean[:, None] + grand_mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    positive = eigenvalues > max(eigenvalues[0], 1.0) * 1e-12
    if positive.sum() < n_components:
        raise ValueError("Requested more components than positive kernel eigenvalues")
    return {
        "training_data": X.copy(), "gamma": gamma,
        "column_mean": column_mean, "grand_mean": grand_mean,
        "projection": eigenvectors[:, :n_components] / np.sqrt(eigenvalues[:n_components]),
    }


def transform_kernel_pca(X_new: np.ndarray, fitted) -> np.ndarray:
    """Centre each new kernel row using the stored training reference."""
    K_new = kernel_matrix(X_new, fitted["training_data"], fitted["gamma"])
    centered = (K_new - K_new.mean(axis=1, keepdims=True)
                - fitted["column_mean"][None, :] + fitted["grand_mean"])
    return centered @ fitted["projection"]


def kernel_pca(X: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Convenience fit-transform on the training observations."""
    fitted = fit_kernel_pca(X, gamma)
    return transform_kernel_pca(X, fitted)


def main() -> None:
    X, y = data()

    Z_primal = pca_primal(X)[:, :2]
    Z_dual = pca_dual(X)[:, :2]
    # The primal and the dual form must give the same scores (up to the sign
    # of an eigenvector). This holds only with the 1/sqrt(n*lambda) scaling
    # of alpha: `pca_dual` divides by sqrt of the eigenvalue of X X^T, which
    # is n*lambda with lambda the eigenvalue of the covariance (1/n) X^T X.
    for k in range(2):
        if Z_primal[:, k] @ Z_dual[:, k] < 0:      # eigenvector sign is arbitrary
            Z_dual[:, k] = -Z_dual[:, k]
    print("max |primal - dual| score difference:",
          f"{np.abs(Z_primal - Z_dual).max():.2e}")
    assert np.allclose(Z_primal, Z_dual), "primal and dual PCA scores differ"

    # Check new-sample centring on held-out points, not just training scores.
    held_out = np.arange(len(X)) % 5 == 0
    train, new = X[~held_out], X[held_out]
    fitted = fit_kernel_pca(train)
    reference = KernelPCA(n_components=2, kernel="rbf", gamma=GAMMA,
                          eigen_solver="dense").fit(train)
    our_train = transform_kernel_pca(train, fitted)
    reference_train = reference.transform(train)
    signs = np.where(np.sum(our_train * reference_train, axis=0) < 0, -1, 1)
    our_new = transform_kernel_pca(new, fitted) * signs
    reference_new = reference.transform(new)
    assert np.allclose(our_train * signs, reference_train, atol=1e-8, rtol=1e-8)
    assert np.allclose(our_new, reference_new, atol=1e-8, rtol=1e-8)
    # A point's coordinates must not depend on other members of its new batch.
    assert np.allclose(transform_kernel_pca(new[:1], fitted)[0],
                       transform_kernel_pca(new, fitted)[0], atol=1e-12, rtol=1e-12)
    print("max |held-out kernel PCA - sklearn| score difference:",
          f"{np.abs(our_new - reference_new).max():.2e}")

    scores = [
        ("original data", X),
        ("PCA — primal form", Z_primal),
        ("PCA — dual form", Z_dual),
        ("kernel PCA (RBF), from scratch", kernel_pca(X)),
        ("kernel PCA (RBF), scikit-learn",
         KernelPCA(n_components=2, kernel="rbf", gamma=GAMMA).fit_transform(X)),
    ]

    fig, axes = plt.subplots(1, len(scores), figsize=(4 * len(scores), 4))
    for ax, (title, Z) in zip(axes, scores):
        ax.scatter(Z[:, 0], Z[:, 1], c=y, s=10, cmap="coolwarm")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
