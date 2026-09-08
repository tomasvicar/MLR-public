"""Konjugovaný prior: Bernoulliho věrohodnost a Beta prior.

Ukázka k jedenácté přednášce (Probabilistic models 1). Prior Beta(a, b)
se po n pozorováních Bernoulliho veličiny mění zase na Beta:

    a' = a + sum(x_i),      b' = b + n - sum(x_i)

Ze stejných dat vyjdou tři různá čísla: MLE, MAP a střední hodnota
aposteriorního rozdělení (= posterior predictive p(x = 1)).

    MPLBACKEND=Agg uv run python conjugate_prior_beta.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

data = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])     # 7 uspechu z 10
n, k = len(data), int(data.sum())
a, b = 2, 2                                          # prior Beta(2, 2)
a1, b1 = a + k, b + n - k                            # posterior

theta_mle = k / n
theta_map = (k + a - 1) / (n + a + b - 2)
prediktivni = a1 / (a1 + b1)

print(f"data: n = {n}, sum(x) = {k}")
print(f"prior     Beta({a}, {b})")
print(f"posterior Beta({a1}, {b1})")
print(f"theta_MLE                = {theta_mle:.4f}")
print(f"theta_MAP                = {theta_map:.4f}")
print(f"posterior predictive p(x=1) = {prediktivni:.4f}")
assert (a1, b1) == (9, 5)

# aktualizace po jednom vzorku dá totéž jako naráz — to je celý smysl
# konjugovaného prioru
aa, bb = a, b
for x in data:
    aa, bb = aa + x, bb + (1 - x)
assert (aa, bb) == (a1, b1)

# --- „black swan paradox“: MLE po samých nulách -------------------------
print("\nblack swan paradox (same prior, only zeros observed):")
for m in (1, 5, 20):
    print(f"  n = {m:2d}:  MLE = {0.0:.3f}   "
          f"MAP = {(0 + a - 1) / (m + a + b - 2):.3f}")

# --- graf ----------------------------------------------------------------
theta = np.linspace(1e-4, 1 - 1e-4, 500)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(theta, stats.beta.pdf(theta, a, b), color="#8895a0",
        label=f"prior Beta({a},{b})")
ax.plot(theta, stats.beta.pdf(theta, k + 1, n - k + 1), "--",
        color="#e9a23b", label="likelihood (normalised)")
ax.plot(theta, stats.beta.pdf(theta, a1, b1), color="#007f86", linewidth=2.2,
        label=f"posterior Beta({a1},{b1})")
ax.axvline(theta_mle, color="#e9a23b", linestyle=":")
ax.axvline(theta_map, color="#007f86", linestyle=":")
ax.set(xlabel=r"$\theta$", ylabel="density", xlim=(0, 1),
       title=f"MLE = {theta_mle:.3f},  MAP = {theta_map:.3f},  "
             f"predictive = {prediktivni:.3f}")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
