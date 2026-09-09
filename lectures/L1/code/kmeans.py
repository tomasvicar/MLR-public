"""Lloyd's k-means algorithm from scratch and with scikit-learn.

Two alternating steps: assign every point to the nearest centroid, then move
every centroid to the mean of its points. The inertia J can only decrease.
Run with ``uv run python lectures/L1/code/kmeans.py``.
"""

import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(42)
data = np.vstack([
    rng.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 10),
    rng.multivariate_normal([8, 8], [[0.5, 0], [0, 0.5]], 20),
    rng.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], 30),
])

k = 3
max_iterations = 20

# --- by hand -----------------------------------------------------------
centroids = data[rng.choice(len(data), k, replace=False)].copy()

for iteration in range(max_iterations):
    # 1) assignment: nearest centroid for every point
    distances = np.sqrt(((data - centroids[:, None]) ** 2).sum(axis=2))
    assignments = distances.argmin(axis=0)

    # inertia = the objective the algorithm minimizes
    inertia = sum(((data[assignments == j] - centroids[j]) ** 2).sum() for j in range(k))
    print(f"iteration {iteration:2d}   J = {inertia:8.3f}")

    # 2) update: centroid = mean of its points
    new_centroids = np.array([data[assignments == j].mean(axis=0)
                              if np.any(assignments == j) else centroids[j]
                              for j in range(k)])
    if np.allclose(new_centroids, centroids):
        print("converged — no centroid moved")
        break
    centroids = new_centroids

print("\ncentroids found by hand:")
print(np.round(centroids[np.lexsort(centroids.T)], 3))

# --- the same with scikit-learn ----------------------------------------
model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data)
print("\ncentroids found by scikit-learn:")
print(np.round(model.cluster_centers_[np.lexsort(model.cluster_centers_.T)], 3))
print(f"scikit-learn inertia: {model.inertia_:.3f}")
