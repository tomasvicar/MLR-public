"""Logistická regrese gradientním vzestupem — a její vztah ke cross-entropy.

Ukázka ke dvanácté přednášce (Probabilistic models 2). Data jsou plně
syntetická, generátor je převzatý ze cvičení `logistic.ipynb`: hodiny
učení denně, počet cvičných testů a počet piv den před zkouškou -> složil.

Trénuje se přesně podle vzorce ze slidu (gradientní VZESTUP na
log-věrohodnosti, bias absorbovaný do w jako nultý sloupec jedniček):

    w^(j) <- w^(j) + mu * suma_i (y_i - sigma(w^T x_i)) x_i^(j)

Skript ukazuje tři věci:
  1. log-věrohodnost monotónně roste,
  2. L_log = -n * BCE, tedy maximalizace věrohodnosti = minimalizace
     binární cross-entropy z šesté přednášky,
  3. výsledek sedí se `sklearn.linear_model.LogisticRegression`
     bez regularizace.

    MPLBACKEND=Agg uv run python logistic_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- data (generátor ze cvičení) -----------------------------------------
rng = np.random.default_rng(0)
n = 500
hodiny = np.clip(rng.normal(3, 1.5, n), 0, 8)
testy = rng.integers(0, 6, n).astype(float)
piva = rng.integers(0, 10, n).astype(float)
X = np.column_stack([hodiny, testy, piva])

beta = np.array([-2.0, 1.2, 0.4, -0.9])          # skryté "pravé" parametry
p_pravé = 1 / (1 + np.exp(-(beta[0] + X @ beta[1:])))
y = rng.binomial(1, p_pravé).astype(float)


def sigmoida(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_verohodnost(w, Xb, y):
    """L_log = suma_i y_i log sigma(w^T x_i) + (1 - y_i) log(1 - sigma(w^T x_i))"""
    p = sigmoida(Xb @ w)
    return np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


def binarni_cross_entropy(w, Xb, y):
    """BCE = -1/n suma_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]  (šestá přednáška)"""
    p = sigmoida(Xb @ w)
    return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


# --- gradientní vzestup ---------------------------------------------------
Xb = np.column_stack([np.ones(n), X])            # bias v nultém sloupci
w = np.zeros(Xb.shape[1])
mu = 0.0005
iteraci = 15000

historie = []
for _ in range(iteraci):
    p = sigmoida(Xb @ w)
    historie.append(log_verohodnost(w, Xb, y))
    w = w + mu * (Xb.T @ (y - p))                # gradientní VZESTUP

historie = np.array(historie)
assert np.all(np.diff(historie) > -1e-9), "log-věrohodnost musí růst"

print("odhadnute vahy (gradientni vzestup):")
for jmeno, hodnota, pravda in zip(["w0", "hours", "tests", "beers"], w, beta):
    print(f"  {jmeno:>6s} = {hodnota:7.4f}   (skutecna hodnota {pravda:5.2f})")
presnost = ((sigmoida(Xb @ w) >= 0.5).astype(float) == y).mean()
print(f"presnost na trenovaci mnozine: {presnost:.3f}")

# --- log-věrohodnost versus binární cross-entropy -------------------------
L = log_verohodnost(w, Xb, y)
BCE = binarni_cross_entropy(w, Xb, y)
print(f"\nL_log = {L:.4f}")
print(f"BCE   = {BCE:.6f}    -n * BCE = {-n * BCE:.4f}")
assert abs(L - (-n * BCE)) < 1e-6
print("=> maximalizace log-verohodnosti = minimalizace binarni cross-entropy")

# --- kontrola proti scikit-learn -----------------------------------------
lr = LogisticRegression(C=np.inf, max_iter=10000).fit(X, y)
w_sklearn = np.r_[lr.intercept_, lr.coef_.ravel()]
print(f"\nsklearn: {np.round(w_sklearn, 4)}")
print(f"maximalni rozdil vah: {np.max(np.abs(w - w_sklearn)):.4f}")
assert np.max(np.abs(w - w_sklearn)) < 1e-3

# --- interpretace vah: odds ratio ----------------------------------------
print("\nodds ratio OR_j = exp(w_j):")
for jmeno, hodnota in zip(["hours", "tests", "beers"], w[1:]):
    OR = np.exp(hodnota)
    p0 = 0.10                                     # priklad ze slidu
    odds0 = p0 / (1 - p0)
    odds1 = odds0 * OR
    print(f"  {jmeno:>6s}: OR = {OR:5.3f}   p = 10 % -> odds {odds0:.4f}"
          f" -> {odds1:.4f} -> p = {odds1 / (1 + odds1) * 100:5.2f} %")

# --- obrázek --------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.5))
a1.plot(np.arange(1, iteraci + 1), historie, color="#c73e1d", linewidth=2)
a1.set(xscale="log", xlabel="iteration (log scale)",
       ylabel="$\\mathcal{L}_{log}$", title="gradient ascent")
a1.grid(alpha=0.25)

mrizka = np.linspace(0, 8, 300)
stred = X.mean(axis=0)
z = w[0] + w[1] * mrizka + w[2] * stred[1] + w[3] * stred[2]
a2.scatter(X[:, 0], y + rng.normal(0, 0.02, n), s=10, color="#8895a0", alpha=0.6)
a2.plot(mrizka, sigmoida(z), color="#c73e1d", linewidth=2.4)
a2.set(xlabel="hours of study per day", ylabel="$p(y = 1 | x)$",
       title="fitted sigmoid (other features at their mean)")
fig.tight_layout()
plt.show()
