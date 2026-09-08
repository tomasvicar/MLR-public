"""EM algoritmus pro Gaussian mixture model — od nuly, ~30 řádků numpy.

Ukázka ke dvanácté přednášce (Probabilistic models 2). E-krok a M-krok
jsou napsané přesně podle vzorců ze slidů:

    E:  q(x_i; theta_k) = w_k p(x_i; theta_k) / suma_j w_j p(x_i; theta_j)

    M:  w_k     = 1/n suma_i q(x_i; theta_k)
        mu_k    = suma_i q(x_i; theta_k) x_i / suma_i q(x_i; theta_k)
        Sigma_k = suma_i q(x_i; theta_k) (x_i - mu_k)(x_i - mu_k)^T
                  / suma_i q(x_i; theta_k)

kde theta_k = [mu_k, Sigma_k], stejne jako ve slidech.

Pozor na tvar posledního vzorce: je to VNĚJŠÍ součin, výsledek musí být
matice d x d. (Slide o vícerozměrné Gaussovce má transpozici u špatného
činitele, viz `sources.md`.)

Data i inicializace jsou stejné jako ve cvičení
`ex13_probabilistic_models_2.ipynb`: seed 801, generující rozdělení se středy
(1, 2) a (-1, -2), inicializace mu_1 = x_1, mu_2 = x_2, Sigma_k = I, w_k = 0,5.

    MPLBACKEND=Agg uv run python em_gmm.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

# --- data (generátor ze cvičení) -----------------------------------------
np.random.seed(801)
mu1_pravé, sigma1_pravé, m1 = np.array([1, 2]), np.array([[1, 0.2], [0.2, 1]]), 100
mu2_pravé, sigma2_pravé, m2 = np.array([-1, -2]), np.array([[3, 0], [0, 1]]), 50
X1 = np.random.randn(m1, 2) @ np.linalg.cholesky(sigma1_pravé) + mu1_pravé
X2 = np.random.randn(m2, 2) @ np.linalg.cholesky(sigma2_pravé) + mu2_pravé
X = np.vstack([X1, X2])
n, d = X.shape
K = 2


def hustota(X, mu, sigma):
    """p(x; mu, Sigma) — vícerozměrná normální hustota."""
    R = X - mu
    Sinv = np.linalg.inv(sigma)
    kvadrat = np.sum(R @ Sinv * R, axis=1)
    return np.exp(-0.5 * kvadrat) / ((2 * np.pi) ** (d / 2)
                                     * np.sqrt(np.linalg.det(sigma)))


# --- 1) inicializace ------------------------------------------------------
mus = [X[0].copy(), X[1].copy()]
sigmy = [np.eye(d), np.eye(d)]
w = np.full(K, 1.0 / K)

log_verohodnosti = []
for iterace in range(40):
    # --- 2) E-krok --------------------------------------------------------
    hustoty = np.column_stack([hustota(X, mus[k], sigmy[k]) for k in range(K)])
    vazene = w * hustoty
    smes = vazene.sum(axis=1)
    q = vazene / smes[:, None]
    assert np.allclose(q.sum(axis=1), 1)

    log_verohodnosti.append(np.sum(np.log(smes)))

    # --- 3) M-krok --------------------------------------------------------
    for k in range(K):
        Nk = q[:, k].sum()
        w[k] = Nk / n
        mus[k] = (q[:, k] @ X) / Nk
        R = X - mus[k]
        sigmy[k] = (R * q[:, k][:, None]).T @ R / Nk   # vnější součin

    # --- 4) stopovací kritérium ------------------------------------------
    if len(log_verohodnosti) > 1 and \
            log_verohodnosti[-1] - log_verohodnosti[-2] < 1e-6:
        break

log_verohodnosti = np.array(log_verohodnosti)
assert np.all(np.diff(log_verohodnosti) > -1e-9), "log-věrohodnost musí růst"

print(f"konvergence po {len(log_verohodnosti)} iteracich")
for k in range(K):
    print(f"  slozka {k+1}: w = {w[k]:.3f}   mu = {np.round(mus[k], 3)}")
    print(f"             Sigma =\n{np.round(sigmy[k], 3)}")
print(f"log L = {log_verohodnosti[-1]:.4f}")

# --- kontrola proti scikit-learn -----------------------------------------
gm = GaussianMixture(n_components=K, covariance_type="full", n_init=10,
                     random_state=0).fit(X)
print(f"log L (sklearn) = {gm.score(X) * n:.4f}")
assert abs(log_verohodnosti[-1] - gm.score(X) * n) < 0.05


# --- obrázek --------------------------------------------------------------
def elipsa(ax, mu, sigma, barva, nsigma=2.0):
    """Elipsa z vlastních čísel kovarianční matice."""
    vals, vecs = np.linalg.eigh(sigma)
    uhel = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(mu, *(2 * nsigma * np.sqrt(vals)), angle=uhel,
                         color=barva, alpha=0.25))


fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
a1.scatter(X[:, 0], X[:, 1], c=q[:, 0], cmap="coolwarm", s=26, marker="+",
           vmin=0, vmax=1)
for k, barva in enumerate(["#c73e1d", "#12355b"]):
    elipsa(a1, mus[k], sigmy[k], barva)
a1.set(xlabel="$x_1$", ylabel="$x_2$",
       title="EM result — colour is $q(\\mathbf{x}_i; \\boldsymbol{\\theta}_1)$")

a2.plot(log_verohodnosti, "o-", color="#c73e1d", markersize=4)
a2.axhline(gm.score(X) * n, linestyle="--", color="#12355b",
           label="sklearn GaussianMixture")
a2.set(xlabel="iteration", ylabel="$\\log L$", title="log-likelihood grows monotonically")
a2.legend(frameon=False)
fig.tight_layout()
plt.show()
