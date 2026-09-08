"""Naive Bayes classifier od nuly — a porovnání se scikit-learn.

Ukázka ke dvanácté přednášce (Probabilistic models 2). Dva příznaky
s různým typem rozdělení, tři třídy:

    x_1 ... velikost nádoru [mm]   -> normální rozdělení N(mu, sigma^2)
    x_2 ... HPV pozitivita {0, 1}  -> Bernoulliho rozdělení Bern(theta)

Naivní předpoklad znamená, že místo jednoho dvourozměrného rozdělení
p(x_1, x_2 | y_k) fitujeme K * d = 3 * 2 = 6 jednorozměrných rozdělení
a jejich součin. Parametry se odhadují MLE, apriorní pravděpodobnosti
tříd relativní četností.

Data jsou **datová sada ze cvičení** `ex13_probabilistic_models_2.ipynb`
(`ex12_files/Kopie souboru dataset.csv`, 203 vzorků) — táž, se kterou budete
pracovat na cvičení, takže čísla na slidech i tady sedí s tím, co vám vyjde.

    MPLBACKEND=Agg uv run python naive_bayes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB

K = 3  # počet tříd
D = 2  # počet příznaků

# --- data ----------------------------------------------------------------
# Adresář exportu Google Disku má v názvu časové razítko, proto glob.
REPO = Path(__file__).resolve().parents[3]
CESTA = sorted(REPO.glob("google_disk_2025/*/2025/ex12_files/Kopie souboru dataset.csv"))
if not CESTA:
    raise FileNotFoundError("dataset.csv ze cvičení ex12_files/ nenalezen")

radky = CESTA[0].read_text(encoding="utf-8-sig").strip().splitlines()
velikost, hpv, y = [], [], []
for radek in radky[1:]:
    a, b, c = radek.split(";")
    velikost.append(float(a))
    hpv.append(1.0 if b.strip().upper() == "P" else 0.0)
    y.append(int(c))
X = np.column_stack([velikost, hpv])
y = np.array(y)

# Dělení jako ve cvičení: náhodná permutace, dvě třetiny na trénink.
poradi = np.random.default_rng(0).permutation(len(X))
delici = 2 * len(X) // 3
trenovaci, testovaci = poradi[:delici], poradi[delici:]
X_tr, y_tr = X[trenovaci], y[trenovaci]
X_te, y_te = X[testovaci], y[testovaci]


# --- odhad parametrů (M-krok naivního Bayese je prostě MLE) --------------
def fit(X, y):
    prior = np.array([(y == k).mean() for k in range(K)])
    mu = np.array([X[y == k, 0].mean() for k in range(K)])
    sigma = np.array([X[y == k, 0].std(ddof=0) for k in range(K)])
    theta = np.array([X[y == k, 1].mean() for k in range(K)])
    return prior, mu, sigma, theta


def normalni(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def bernoulli(x, theta):
    return theta ** x * (1 - theta) ** (1 - x)


def posterior(vzorky, model, priznaky=(0, 1)):
    """p(y_k | x) = p(y_k) * soucin_i p(x_i | y_k), normalizovane na 1."""
    prior, mu, sigma, theta = model
    vzorky = np.atleast_2d(vzorky)
    spolecne = np.tile(prior, (len(vzorky), 1))
    if 0 in priznaky:
        spolecne = spolecne * normalni(vzorky[:, [0]], mu, sigma)
    if 1 in priznaky:
        spolecne = spolecne * bernoulli(vzorky[:, [1]], theta)
    return spolecne / spolecne.sum(axis=1, keepdims=True)


model = fit(X_tr, y_tr)
prior, mu, sigma, theta = model

print("odhadnuté parametry (MLE na trénovací množině)")
for k in range(K):
    print(f"  trida y_{k+1}:  p(y_k) = {prior[k]:.3f}   "
          f"mu = {mu[k]:6.2f}  sigma = {sigma[k]:5.2f}   theta = {theta[k]:.3f}")

p_te = posterior(X_te, model)
presnost = (p_te.argmax(1) == y_te).mean()
print(f"\npresnost na testovaci mnozine: {presnost:.3f}")
assert abs(presnost - 0.647) < 5e-4, presnost   # cislo, ktere je na slidu

# --- naivní předpoklad: jak moc je v těchhle datech porušený? ------------
# Podmíněná nezávislost neznamená nezávislost: dohromady jsou příznaky
# korelované, uvnitř tříd skoro ne.  (Čísla jsou na slidu o naivním předpokladu.)
celkova = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
v_tridach = [float(np.corrcoef(X[y == k, 0], X[y == k, 1])[0, 1]) for k in range(K)]
print("\nkorelace velikosti a HPV:")
print(f"  pres vsechny tridy dohromady: {celkova:.2f}")
print("  uvnitr trid:                  "
      + "  ".join(f"{c:.2f}" for c in v_tridach))
assert abs(celkova - 0.16) < 5e-3, celkova
assert np.allclose(np.round(v_tridach, 2), [0.03, 0.05, 0.09]), v_tridach
assert all(abs(c) < celkova for c in v_tridach)

# --- posteriory tří konkrétních vzorků -----------------------------------
print("\nposteriory p(y_k | x) — sloupce y_1, y_2, y_3")
for vzorek in [np.array([7.0, 0.0]), np.array([15.0, 0.0]), np.array([15.0, 1.0])]:
    p = posterior(vzorek, model)[0]
    hpv_popis = "HPV+" if vzorek[1] else "HPV-"
    print(f"  velikost {vzorek[0]:5.1f} mm, {hpv_popis}:  "
          f"{p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}   (suma {p.sum():.6f})")
    assert abs(p.sum() - 1) < 1e-12

# --- chybějící příznak: ze součinu se prostě vynechá ---------------------
chybi = np.array([15.0, 1.0])
print("\nchybejici priznak — do soucinu vstoupi jen ten, ktery mame:")
print(f"  oba priznaky:      {np.round(posterior(chybi, model)[0], 3)}")
print(f"  jen velikost:      {np.round(posterior(chybi, model, (0,))[0], 3)}")
print(f"  jen HPV:           {np.round(posterior(chybi, model, (1,))[0], 3)}")

# --- kontrola proti scikit-learn ----------------------------------------
# sklearn nemá klasifikátor s různým rozdělením na různých příznacích,
# takže se složí ze dvou: log p(y_k) + log p(x_1|y_k) + log p(x_2|y_k).
gnb = GaussianNB().fit(X_tr[:, [0]], y_tr)
bnb = BernoulliNB(alpha=1e-12).fit(X_tr[:, [1]], y_tr)
# predict_log_proba uz prior obsahuje, takze pri scitani dvou modelu
# se jeden prior musi odecist:  log p(y|x1) + log p(y|x2) - log p(y)
log_prior = np.log(prior)
log_spolecne = (gnb.predict_log_proba(X_te[:, [0]])
                + bnb.predict_log_proba(X_te[:, [1]]) - log_prior)
p_sklearn = np.exp(log_spolecne - log_spolecne.max(1, keepdims=True))
p_sklearn /= p_sklearn.sum(1, keepdims=True)
print(f"\nmaximalni rozdil proti sklearn: {np.abs(p_sklearn - p_te).max():.2e}")
assert np.abs(p_sklearn - p_te).max() < 1e-6  # GaussianNB pricita var_smoothing

# --- obrázek -------------------------------------------------------------
barvy = ["#3a8a52", "#e9a23b", "#c73e1d"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
mrizka = np.linspace(0, 28, 400)
for k in range(K):
    a1.hist(X_tr[y_tr == k, 0], bins=np.arange(0, 28.1, 2.0), density=True,
            color=barvy[k], alpha=0.35)
    a1.plot(mrizka, normalni(mrizka, mu[k], sigma[k]), color=barvy[k], linewidth=2,
            label=f"$y_{k+1}$")
a1.set(xlabel="tumour size [mm]", ylabel="$p(x_1 | y_k)$")
a1.legend(frameon=False)

for k in range(K):
    a2.plot(mrizka, posterior(np.column_stack([mrizka, np.zeros_like(mrizka)]),
                              model)[:, k], color=barvy[k], linewidth=2,
            label=f"$p(y_{k+1} | x)$, HPV$-$")
    a2.plot(mrizka, posterior(np.column_stack([mrizka, np.ones_like(mrizka)]),
                              model)[:, k], color=barvy[k], linewidth=2,
            linestyle="--")
a2.set(xlabel="tumour size [mm]", ylabel="posterior", ylim=(0, 1.05))
a2.legend(frameon=False, fontsize=8)
a2.set_title("solid = HPV$-$, dashed = HPV$+$")
fig.tight_layout()
plt.show()
