"""Bayesovo pravidlo: pravděpodobnost nemoci po pozitivním testu.

Ukázka k jedenácté přednášce (Probabilistic models 1) — příklad 2 ze slidů.
Test má senzitivitu i specificitu 98 %; hledá se p(A | B+), tedy PPV
(positive predictive value) z druhé přednášky. Pointa: stejný test dá
při prevalenci 10 % výsledek 84 %, ale při prevalenci 1 % jen 33 %.

    MPLBACKEND=Agg uv run python bayes_prevalence.py
"""

import matplotlib.pyplot as plt
import numpy as np

SENZITIVITA = 0.98      # p(B+ | A)
SPECIFICITA = 0.98      # p(B- | not A), takze p(B+ | not A) = 0.02


def ppv(prevalence: float, se: float = SENZITIVITA,
        sp: float = SPECIFICITA) -> tuple[float, float]:
    """Vrátí (p(B+), p(A | B+)) — zákon úplné pravděpodobnosti a Bayes."""
    p_bplus = se * prevalence + (1 - sp) * (1 - prevalence)
    return p_bplus, se * prevalence / p_bplus


for prevalence in (0.10, 0.01):
    p_bplus, p_a_dano_b = ppv(prevalence)
    print(f"prevalence {prevalence:5.1%}:  p(B+) = {p_bplus:.4f}   "
          f"p(A|B+) = {p_a_dano_b:.4f}  ->  {round(p_a_dano_b * 100):d} %")

# kontrola proti číslům ze slidů
assert abs(ppv(0.10)[0] - 0.116) < 1e-9
assert abs(ppv(0.01)[0] - 0.0296) < 1e-9
assert round(ppv(0.10)[1] * 100) == 84 and round(ppv(0.01)[1] * 100) == 33

# --- ověření simulací ----------------------------------------------------
rng = np.random.default_rng(11)
n = 2_000_000
nemocny = rng.random(n) < 0.01
test = np.where(nemocny, rng.random(n) < SENZITIVITA,
                rng.random(n) >= SPECIFICITA)
print(f"\nsimulace ({n} lidi, prevalence 1 %): "
      f"p(A|B+) = {nemocny[test].mean():.4f}")

# --- PPV jako funkce prevalence -----------------------------------------
prevalence = np.linspace(0.001, 0.5, 500)
krivka = np.array([ppv(p)[1] for p in prevalence])

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(prevalence * 100, krivka * 100, color="#007f86", linewidth=2)
for p, barva in ((0.01, "#c73e1d"), (0.10, "#12355b")):
    ax.plot(p * 100, ppv(p)[1] * 100, "o", color=barva)
ax.set(xlabel="prevalence p(A) [%]", ylabel="p(A | B+) = PPV [%]",
       xlim=(0, 50), ylim=(0, 100),
       title="the same test, different prevalence")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
