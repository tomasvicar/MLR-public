"""UPGMA (average-linkage hierarchical clustering) from scratch and with SciPy.

In every step the two closest clusters are merged and the distance matrix is
updated with the size-weighted average of the two merged rows.
Run with ``uv run python lectures_latex_2025/L1/code/upgma.py``.
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

names = ["A", "B", "C", "D"]
X = np.array([[0.0, 0.0], [1.1, 0.2], [0.2, 1.3], [1.3, 1.1]])

# pairwise Euclidean distance matrix
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

clusters = [(n,) for n in names]
sizes = {(n,): 1 for n in names}
labels = list(names)
matrix = D.copy()

print("initial distance matrix:")
print("     " + "  ".join(f"{n:>5}" for n in labels))
for i, n in enumerate(labels):
    print(f"{n:>4} " + "  ".join(f"{v:5.2f}" for v in matrix[i]))

while len(clusters) > 1:
    # find the closest pair (ignore the zero diagonal)
    working = np.where(np.eye(len(clusters), dtype=bool), np.inf, matrix)
    i, j = np.unravel_index(working.argmin(), working.shape)
    i, j = sorted((i, j))
    height = matrix[i, j]

    ni, nj = sizes[clusters[i]], sizes[clusters[j]]
    print(f"\nmerge {''.join(clusters[i])} + {''.join(clusters[j])} at d = {height:.2f}")

    # size-weighted average of the two rows — this is the UPGMA update rule
    new_row = (ni * matrix[i] + nj * matrix[j]) / (ni + nj)
    matrix = np.vstack([matrix, new_row])
    matrix = np.column_stack([matrix, np.append(new_row, 0.0)])
    keep = [k for k in range(len(clusters) + 1) if k not in (i, j)]
    matrix = matrix[np.ix_(keep, keep)]

    merged = tuple(sorted(clusters[i] + clusters[j]))
    sizes[merged] = ni + nj
    clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    labels = ["".join(c) for c in clusters]

    print("     " + "  ".join(f"{n:>5}" for n in labels))
    for r, n in enumerate(labels):
        print(f"{n:>4} " + "  ".join(f"{v:5.2f}" for v in matrix[r]))

# --- the same with SciPy ----------------------------------------------
Z = linkage(X, method="average")   # 'average' linkage == UPGMA
print("\nSciPy linkage matrix (child, child, distance, size):")
print(np.round(Z, 3))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dendrogram(Z, labels=names)
    plt.ylabel("Distance")
    plt.title("Dendrogram (UPGMA)")
    plt.show()
