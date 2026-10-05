"""How nu and the choice of kernel change the nu-SVM decision boundary.

An interactive version of the figures from the "SVM visualizations" slide — just
rewrite `separable` and the model parameters and run:

    uv run python lectures/L4/code/svm_visualization.py

Both the data and the plotting come from the notebook `SVM_visualisation.ipynb`.
The coloured surface is the predicted probability, the solid line is the
boundary, the dashed lines are the margin edges and the circles mark the support
vectors.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import NuSVC


def data(separable):
    """0 = overlapping classes, 1 = linearly separable."""
    if separable:
        return make_classification(n_samples=50, n_features=2, n_informative=2,
                                   n_redundant=0, n_classes=2, class_sep=1,
                                   n_clusters_per_class=1, random_state=49)
    return make_classification(n_samples=50, n_features=2, n_informative=2,
                               n_redundant=0, n_classes=2, class_sep=0.8,
                               n_clusters_per_class=1, random_state=45)


def plot_model(ax, model, features, labels):
    f1, f2 = np.meshgrid(
        np.linspace(features[:, 0].min() - 1, features[:, 0].max() + 1, 100),
        np.linspace(features[:, 1].min() - 1, features[:, 1].max() + 1, 100))
    grid = np.stack((f1.ravel(), f2.ravel()), axis=1)

    ax.contourf(f1, f2, model.predict_proba(grid)[:, 1].reshape(f1.shape),
                alpha=0.4)
    ax.contour(f1, f2, model.decision_function(grid).reshape(f1.shape),
               colors="k", levels=[-1, 0, 1], linestyles=["--", "-", "--"])
    ax.scatter(features[:, 0], features[:, 1], c=labels, edgecolor="k",
               linewidths=0.6)
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors="none", edgecolors="k")


def main():
    variants = [
        (1, dict(kernel="linear", nu=0.001)),
        (1, dict(kernel="linear", nu=0.2)),
        (1, dict(kernel="linear", nu=0.5)),
        (0, dict(kernel="rbf", nu=0.4, gamma=0.1)),
        (0, dict(kernel="rbf", nu=0.4, gamma=5)),
        (0, dict(kernel="poly", nu=0.4, degree=3)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (separable, params) in zip(axes.ravel(), variants):
        features, labels = data(separable)
        model = NuSVC(probability=True, random_state=0, **params)
        model.fit(features, labels)
        plot_model(ax, model, features, labels)
        caption = ", ".join(f"{k}={v}" for k, v in params.items())
        ax.set_title(f"{caption}  ({len(model.support_)} SV)", fontsize=11)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
