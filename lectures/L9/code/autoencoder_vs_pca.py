"""A bottleneck on faces: PCA versus an autoencoder — on training faces and on new people.

The "Bottleneck" slides compare the reduction of face features by PCA and by a
neural network. Here both are really computed on the public Olivetti faces
dataset (400 images 64 x 64, 40 people with 10 images each; sklearn downloads
and caches it).

The split is by people: the models are fitted on the first 30 people (300
faces) and tested on the remaining 10 people (100 faces) they have never seen.

All models have an **equally narrow bottleneck**: three numbers per image
instead of 4096.

- PCA is a closed-form solution and is optimal among all **linear**
  compressions of a given dimension (on the data it was fitted to),
- a large autoencoder (decoder 3 -> 64 -> 4096, about 266 thousand weights)
  beats PCA on the training faces, but on the new people it is worse than the
  mean face: it has memorized the 300 training faces — a bottleneck alone does
  not prevent overfitting,
- a small regularized autoencoder (decoder 3 -> 8 -> 4096, weight decay,
  early stopping on 6 validation people) generalizes much better, but with so
  few faces it still does not beat PCA-3 on the new people.

The autoencoder is written directly in NumPy (full batch, Adam, cosine decay of
the learning rate). Sklearn's `MLPRegressor` fails here: with three neurons in
the bottleneck the narrow spot dies and the network learns only the mean face.

    MPLBACKEND=Agg uv run python lectures/L9/code/autoencoder_vs_pca.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

CODE = 3         # size of the bottleneck
STEPS = 4000


def train_autoencoder(X, fit, test, width, validation=None, weight_decay=0.0,
                      steps=STEPS, seed=0, patience=10, every=50):
    """Encoder 4096 -> 3 (linear), decoder 3 -> width -> 4096 (tanh).

    Trained on the faces `fit`; the MSE on `fit` and on `test` is recorded every
    `every` steps. With `validation`, training stops when the validation MSE
    has not improved for `patience` checks and the best weights are kept.
    """
    mu = X[fit].mean(0)
    Xall = X - mu
    Xc = Xall[fit]
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    par = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, CODE)).astype(np.float32),
           "W2": rng.normal(0, 1 / np.sqrt(CODE), (CODE, width)).astype(np.float32),
           "b2": np.zeros(width, np.float32),
           "W3": rng.normal(0, 1 / np.sqrt(width), (width, d)).astype(np.float32),
           "b3": np.zeros(d, np.float32)}

    def reconstruct(p, A):
        return np.tanh(A @ p["W1"] @ p["W2"] + p["b2"]) @ p["W3"] + p["b3"]

    def mse(p, mask):
        return float(np.mean((Xall[mask] - reconstruct(p, Xall[mask])) ** 2))

    m = {k: np.zeros_like(v) for k, v in par.items()}
    v = {k: np.zeros_like(val) for k, val in par.items()}
    curve, best = [], (np.inf, 0, None)
    for t in range(1, steps + 1):
        lr = 0.02 * 0.5 * (1 + np.cos(np.pi * t / steps)) + 5e-4
        z = Xc @ par["W1"]                                   # the bottleneck
        h = np.tanh(z @ par["W2"] + par["b2"])
        E = h @ par["W3"] + par["b3"] - Xc
        dR = (2 * E / E.size).astype(np.float32)
        g = {"W3": h.T @ dR, "b3": dR.sum(0)}
        dh = (dR @ par["W3"].T) * (1 - h ** 2)
        g["W2"] = z.T @ dh
        g["b2"] = dh.sum(0)
        g["W1"] = Xc.T @ (dh @ par["W2"].T)
        for k in par:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            par[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (
                np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if weight_decay and k.startswith("W"):           # decoupled, as in AdamW
                par[k] -= lr * weight_decay * par[k]
        if t % every == 0:
            curve.append((t, mse(par, fit), mse(par, test)))
            if validation is not None:
                val = mse(par, validation)
                if val < best[0]:
                    best = (val, t, {k: x.copy() for k, x in par.items()})
                elif t - best[1] >= patience * every:
                    print(f"  early stop at step {t}, best validation step {best[1]}")
                    break
    if validation is not None:
        par = best[2]
    assert (Xc @ par["W1"]).std(0).min() > 1e-3, "the bottleneck died"
    return reconstruct(par, Xall) + mu, mse(par, fit), mse(par, test), np.array(curve)


def main():
    data = fetch_olivetti_faces(shuffle=False)
    X, s = data.data.astype(np.float32), data.images.shape[1]
    people = np.arange(len(X)) // 10
    train, test = people < 30, people >= 30
    fit, validation = people < 24, (people >= 24) & (people < 30)
    print(f"data: {X.shape[0]} images {s} x {s} = {X.shape[1]} pixels; "
          f"training 30 people ({train.sum()} faces), new 10 people ({test.sum()} faces)")
    mu = X[train].mean(0)
    print(f"mean face alone:  training MSE {np.mean((X[train] - mu) ** 2):.4f}, "
          f"new people {np.mean((X[test] - mu) ** 2):.4f}")

    for n in (CODE, 6, 9):
        pca = PCA(n_components=n, random_state=0).fit(X[train])
        R = pca.inverse_transform(pca.transform(X))
        print(f"PCA-{n}:            training MSE {np.mean((X[train] - R[train]) ** 2):.4f}, "
              f"new people {np.mean((X[test] - R[test]) ** 2):.4f}")
    pca = PCA(n_components=CODE, random_state=0).fit(X[train])
    R_pca = pca.inverse_transform(pca.transform(X))

    print("training the large autoencoder (about a minute) …")
    R_big, tr, te, curve_big = train_autoencoder(X, train, test, width=64)
    print(f"AE 3-64-4096:     training MSE {tr:.4f}, new people {te:.4f}  <- overfitted")
    print("training the small regularized autoencoder …")
    R_small, tr_s, te_s, curve_small = train_autoencoder(
        X, fit, test, width=8, validation=validation, weight_decay=0.1)
    print(f"AE 3-8-4096 + wd: training MSE {tr_s:.4f} (24 people), new people {te_s:.4f}")

    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.06, 0.12, 0.36, 0.8])
    for c, colour, name in ((curve_big, "C3", "AE 3-64-4096"),
                            (curve_small, "C0", "AE 3-8-4096, regularized")):
        ax.plot(c[:, 0], c[:, 1], "--", color=colour)
        ax.plot(c[:, 0], c[:, 2], "-", color=colour, label=name + " (solid: new people)")
    ax.axhline(np.mean((X[test] - R_pca[test]) ** 2), color="k", label="PCA-3, new people")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE (dashed: training faces)")
    ax.legend(fontsize=8)
    examples = [300, 320, 340, 350, 370, 390]                # new people only
    rows = ((X, "original"), (R_pca, "PCA-3"), (R_big, "AE large"), (R_small, "AE small"))
    for col, i in enumerate(examples):
        for row, (images, label) in enumerate(rows):
            a = fig.add_axes([0.47 + col * 0.088, 0.73 - row * 0.21, 0.085, 0.19])
            a.imshow(images[i].reshape(s, s), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            if col == 0:
                a.set_ylabel(label, fontsize=9)
    fig.suptitle(f"an equally narrow bottleneck ({CODE} numbers per image), "
                 "faces of people not seen in training", fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()
