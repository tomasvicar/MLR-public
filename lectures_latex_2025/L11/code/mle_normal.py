"""Maximum likelihood pro normální rozdělení — analyticky i numericky.

Ukázka k jedenácté přednášce (Probabilistic models 1). Ukazuje, že
uzavřené vzorce

    mu_MLE     = prumer(x)
    sigma2_MLE = prumer((x - mu_MLE)^2)

opravdu maximalizují log-věrohodnost, a že MLE odhad rozptylu je
vychýlený — dělí se n, ne (n-1).

    MPLBACKEND=Agg uv run python mle_normal.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(1072)
x = rng.normal(170.0, 30.0, 50)          # 50 měření lidské výšky [cm]


def log_verohodnost(mu: float, sigma: float, data: np.ndarray) -> float:
    """log p(D | mu, sigma) pro n nezávislých stejně rozdělených vzorků."""
    return np.sum(-np.log(np.sqrt(2 * np.pi * sigma ** 2))
                  - (data - mu) ** 2 / (2 * sigma ** 2))


# --- analytické řešení ---------------------------------------------------
mu_mle = x.mean()
sigma2_mle = ((x - mu_mle) ** 2).mean()

# --- numerické řešení (žádné vzorce, jen optimalizace) -------------------
vysledek = optimize.minimize(
    lambda p: -log_verohodnost(p[0], np.exp(p[1]), x),
    x0=[150.0, np.log(10.0)], method="Nelder-Mead",
    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
mu_num, sigma_num = vysledek.x[0], np.exp(vysledek.x[1])

print(f"n = {len(x)}")
print(f"analyticky:  mu = {mu_mle:8.4f}   sigma = {np.sqrt(sigma2_mle):8.4f}")
print(f"numericky:   mu = {mu_num:8.4f}   sigma = {sigma_num:8.4f}")
print(f"log p(D|MLE) = {log_verohodnost(mu_mle, np.sqrt(sigma2_mle), x):.4f}")
assert abs(mu_num - mu_mle) < 1e-3 and abs(sigma_num ** 2 - sigma2_mle) < 1e-2

# --- MLE odhad rozptylu je vychýlený ------------------------------------
print(f"\nsigma^2 MLE      (delitel n)     = {sigma2_mle:.3f}")
print(f"sigma^2 nestranny (delitel n - 1) = {x.var(ddof=1):.3f}")
opakovani = np.array([
    (lambda v: v.var(ddof=0))(rng.normal(170.0, 30.0, 50)) for _ in range(20000)])
print(f"E[sigma^2 MLE] pres 20 000 opakovani = {opakovani.mean():.1f} "
      f"(spravna hodnota 900, ocekavano 900*(n-1)/n = {900 * 49 / 50:.1f})")

# --- graf log-věrohodnosti ----------------------------------------------
mu_mrizka = np.linspace(150, 190, 400)
krivka = [log_verohodnost(m, np.sqrt(sigma2_mle), x) for m in mu_mrizka]

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(mu_mrizka, krivka, color="#12355b", linewidth=2)
ax.axvline(mu_mle, color="#007f86", linestyle="--")
ax.set(xlabel=r"$\mu$", ylabel=r"$\log p(D\,|\,\mu,\sigma^2)$",
       title=f"log-likelihood, maximum at the sample mean {mu_mle:.2f} cm")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
