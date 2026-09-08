"""MAP proti MLE na příkladu s krevním tlakem (normální rozdělení, známá sigma).

Ukázka k jedenácté přednášce (Probabilistic models 1). Pro konjugovaný
normální prior N(mu_0, sigma_0^2) a věrohodnost N(mu, sigma^2) se známou
sigma je aposteriorní rozdělení opět normální:

    mu'      = w * prumer(x) + (1 - w) * mu_0,   w = n*sigma_0^2 / (n*sigma_0^2 + sigma^2)
    sigma'^2 = (1/sigma_0^2 + n/sigma^2)^(-1)

MAP odhad je konvexní kombinace výběrového průměru (MLE) a apriorní
střední hodnoty; s rostoucím n konverguje k MLE.

    MPLBACKEND=Agg uv run python map_vs_mle.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

data = np.array([121.0, 109.0, 115.0])      # tri mereni TK [mmHg]
sigma = 6.0                                 # znama presnost tonometru
mu_0, sigma_0 = 120.0, 12.0                 # prior: populace


def posterior(x: np.ndarray) -> tuple[float, float]:
    """Vrátí (mu', sigma') aposteriorního rozdělení parametru mu."""
    n = len(x)
    w = n * sigma_0 ** 2 / (n * sigma_0 ** 2 + sigma ** 2)
    mu = w * x.mean() + (1 - w) * mu_0
    var = 1.0 / (1.0 / sigma_0 ** 2 + n / sigma ** 2)
    return mu, np.sqrt(var)


mu_post, sigma_post = posterior(data)
print(f"MLE:  mu = {data.mean():.2f} mmHg")
print(f"MAP:  mu = {mu_post:.2f} mmHg")
print(f"posterior:            N({mu_post:.2f}, {sigma_post ** 2:.2f})"
      f"   -> sigma' = {sigma_post:.2f}")
print(f"posterior predictive: N({mu_post:.2f}, "
      f"{sigma_post ** 2 + sigma ** 2:.2f})"
      f" -> sigma = {np.sqrt(sigma_post ** 2 + sigma ** 2):.2f}")
assert abs(mu_post - 115.3846) < 1e-3 and abs(sigma_post ** 2 - 11.0769) < 1e-3

# --- s rostoucím n MAP konverguje k MLE ---------------------------------
rng = np.random.default_rng(11)
print("\n  n    MLE      MAP     sigma'")
for n in (1, 3, 10, 100, 1000):
    vzorek = rng.normal(115.0, sigma, n)
    m, s = posterior(vzorek)
    print(f"{n:5d} {vzorek.mean():8.2f} {m:8.2f} {s:8.2f}")

# --- graf: prior, věrohodnost, posterior --------------------------------
mu = np.linspace(100, 140, 500)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(mu, stats.norm.pdf(mu, mu_0, sigma_0), color="#8895a0",
        label=f"prior N({mu_0:.0f}, {sigma_0:.0f}$^2$)")
ax.plot(mu, stats.norm.pdf(mu, data.mean(), sigma / np.sqrt(len(data))), "--",
        color="#e9a23b", label="likelihood of $\\mu$ (MLE)")
ax.plot(mu, stats.norm.pdf(mu, mu_post, sigma_post), color="#007f86",
        linewidth=2.2, label=f"posterior N({mu_post:.2f}, "
                             f"{sigma_post:.2f}$^2$)")
ax.plot(data, np.zeros(len(data)), "o", markerfacecolor="none",
        markeredgecolor="#12355b", label="measurements")
ax.set(xlabel=r"$\mu$ [mmHg]", ylabel="density", xlim=(100, 140),
       title="MAP is a compromise between the data and the prior")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
