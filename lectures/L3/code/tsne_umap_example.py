"""Compare PCA, t-SNE and UMAP on the same offline digits dataset.

All methods see all 1797 samples and the same standardized 64 pixel features.
Labels are used only for colouring. The lecture generator imports the fitting
functions below so the runnable example and the slide use identical settings.

    MPLBACKEND=Agg uv run python lectures/L3/code/tsne_umap_example.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

SEED = 42


def embedding_data():
    """Standardize pixels for this exploratory visualization (not a test score)."""
    X, y = load_digits(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def fit_embeddings(X):
    """Use the same input matrix for all three unsupervised fits."""
    return [
        ("PCA", PCA(n_components=2, svd_solver="full").fit_transform(X)),
        ("t-SNE · perplexity=30", TSNE(
            n_components=2, perplexity=30, init="pca", learning_rate="auto",
            max_iter=1000, metric="euclidean", random_state=SEED,
            n_jobs=1).fit_transform(X)),
        ("UMAP · neighbours=15, min_dist=0.1", UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
            random_state=SEED, n_jobs=1).fit_transform(X)),
    ]


def scatter(ax, embedding, labels, title, rasterized=False):
    colors = plt.get_cmap("tab10")
    for digit in range(10):
        mask = labels == digit
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.75,
                   color=colors(digit), label=str(digit), rasterized=rasterized)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    # These are independently scaled maps, not comparable physical coordinates.
    ax.set_aspect("equal", adjustable="datalim")


def main():
    X, y = embedding_data()
    results = fit_embeddings(X)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (title, Z) in zip(axes, results):
        scatter(ax, Z, y, title)
        print(title, "shape:", Z.shape, "finite:", bool(np.isfinite(Z).all()))
    axes[-1].legend(title="digit", markerscale=3, loc="center left",
                    bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Same standardized digits · labels only colour the points · separate axis scales")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
