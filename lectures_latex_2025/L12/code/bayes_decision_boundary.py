"""Bayesovo rozhodovací pravidlo a rozhodovací hranice.

Ukázka ke dvanácté přednášce (Probabilistic models 2). Ukazuje, co dělá
pravidlo ze slidu

    y_k* = argmax_k  p(x | y_k) p(y_k)

když se p(x | y_k) modeluje třemi způsoby:

  1. naivně — součin dvou jednorozměrných normálek (diagonální Sigma_k),
  2. plnou dvourozměrnou normálkou pro každou třídu (kvadratická hranice),
  3. skutečným (generujícím) rozdělením — to je Bayesův optimální
     klasifikátor, lepší už být nejde.

Zároveň je vidět, jak apriorní pravděpodobnost tříd p(y_k) hranici
posouvá: čím vzácnější třída, tím dál od ní hranice leží. To je táž
úvaha jako PPV proti prevalenci ve druhé přednášce.

    MPLBACKEND=Agg uv run python bayes_decision_boundary.py
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(11)

# --- skutečný generující model -------------------------------------------
MU = [np.array([2.4, 2.6]), np.array([5.2, 5.4])]
SIGMA = [np.array([[1.3, 1.0], [1.0, 1.3]]),
         np.array([[1.3, 1.0], [1.0, 1.3]])]
PRIOR = np.array([0.5, 0.5])

n = 400
X = np.vstack([rng.multivariate_normal(MU[k], SIGMA[k], n) for k in range(2)])
y = np.repeat([0, 1], n)


def log_hustota(X, mu, sigma):
    R = X - mu
    znak, logdet = np.linalg.slogdet(sigma)
    return -0.5 * (np.sum(R @ np.linalg.inv(sigma) * R, axis=1) + logdet
                   + 2 * np.log(2 * np.pi))


def log_spolecne(mrizka, rezim, prior):
    """log p(x | y_k) + log p(y_k) pro obě třídy."""
    S = np.zeros((len(mrizka), 2))
    for k in range(2):
        A = X[y == k]
        if rezim == "pravda":
            mu, sigma = MU[k], SIGMA[k]
        else:
            mu = A.mean(axis=0)
            sigma = np.diag(A.var(axis=0)) if rezim == "naivni" \
                else np.cov(A.T, bias=True)
        S[:, k] = log_hustota(mrizka, mu, sigma) + np.log(prior[k])
    return S


xx, yy = np.meshgrid(np.linspace(-2, 10, 320), np.linspace(-2, 10, 320))
mrizka = np.column_stack([xx.ravel(), yy.ravel()])

print("presnost na tychz 800 bodech, na kterych je model natrenovany:")
presnosti = {}
for rezim, popis in [("naivni", "naive Bayes (diagonalni Sigma)"),
                     ("plny", "plna Sigma pro kazdou tridu"),
                     ("pravda", "skutecny model (Bayes optimal)")]:
    predikce = log_spolecne(X, rezim, PRIOR).argmax(axis=1)
    presnosti[rezim] = float((predikce == y).mean())
    print(f"  {popis:32s} {presnosti[rezim]:.3f}")

# Cisla, ktera stoji na slidu "the naive assumption in practice".
assert abs(presnosti["naivni"] - 0.904) < 5e-4, presnosti
assert abs(presnosti["plny"] - 0.896) < 5e-4, presnosti
assert presnosti["naivni"] > presnosti["plny"]   # porusenym predpokladem to netrpi
assert X.min() > -2 and X.max() < 10   # vsech 800 bodu je uvnitr os

# --- obrázek: tři hranice + vliv prioru ----------------------------------
fig, osy = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (rezim, nadpis) in zip(osy, [
        ("naivni", "naive Bayes — independent features"),
        ("plny", "full covariance per class"),
        ("pravda", "true model — Bayes optimal")]):
    Z = np.diff(log_spolecne(mrizka, rezim, PRIOR), axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                colors=["#d9eef7", "#fbe6c4"], alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], colors=["#12355b"], linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, marker="+", color="#12355b",
               alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, marker="x", color="#c73e1d",
               alpha=0.7)
    ax.set(xlabel="$x_1$", title=nadpis, xlim=(-2, 10), ylim=(-2, 10))
osy[0].set_ylabel("$x_2$")

# vliv apriorní pravděpodobnosti tříd na hranici
for prior1, styl in [(0.5, "-"), (0.1, "--"), (0.9, ":")]:
    prior = np.array([1 - prior1, prior1])
    Z = np.diff(log_spolecne(mrizka, "pravda", prior), axis=1).reshape(xx.shape)
    osy[2].contour(xx, yy, Z, levels=[0], colors=["#3a8a52"], linewidths=1.6,
                   linestyles=styl)
osy[2].set_title("true model — $p(y_2) = 0.1 / 0.5 / 0.9$")
fig.tight_layout()
plt.show()
