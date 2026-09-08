"""Distances and vector norms — the four formulas from the slide, by hand.

A distance between two points is simply a norm of their difference vector,
``x = a - b``. Run with ``uv run python lectures_latex_2025/L1/code/distances.py``.
"""

import numpy as np

a = np.array([1.0, 1.0])
b = np.array([2.0, 3.0])
x = a - b

euclidean = np.sqrt(np.sum(x**2))          # L2
manhattan = np.sum(np.abs(x))              # L1
chebyshev = np.max(np.abs(x))              # L-infinity

p = 1.5
minkowski = np.sum(np.abs(x) ** p) ** (1 / p)

cosine = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"a = {a}, b = {b}, x = a - b = {x}")
print(f"Euclidean  ||x||_2      = {euclidean:.4f}")
print(f"Manhattan  ||x||_1      = {manhattan:.4f}")
print(f"Chebyshev  ||x||_inf    = {chebyshev:.4f}")
print(f"Minkowski  ||x||_{p}    = {minkowski:.4f}")
print(f"Cosine distance         = {cosine:.4f}")

# The same with numpy's built-in norm.
print("\nnumpy.linalg.norm:")
for order, name in ((2, "L2"), (1, "L1"), (np.inf, "L-inf"), (p, f"L{p}")):
    print(f"  {name:<6} {np.linalg.norm(x, ord=order):.4f}")
