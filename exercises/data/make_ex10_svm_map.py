"""Precompute the C × gamma map of an RBF SVM on the lab 6 e-bikes for lab 10.

The map is the 5-fold cross-validated accuracy of the same pipeline as in the
notebook (cleaning, median imputation, standardization and one-hot encoding
inside every fold, then `SVC(kernel="rbf")`) on the training trips of lab 6's
split (`test_size=0.25`, stratified, `random_state=42`). The notebook only
loads it, so that Colab does not spend minutes recomputing it.

It also repeats the sampler comparison of the notebook (25 trials, grid /
random / TPE, log and linear ranges) over several study seeds with the real
CV objective, so that the teacher note can quote a mean instead of one run.

Output: `exercises/data/ex10_svm_map.npz` with
    log10_C, log10_gamma   the axes (29 values each)
    cv                     cv[i, j] = CV accuracy at gamma[i], C[j]
    default_cv             SVC() defaults (C = 1, gamma = "scale")
    majority               the majority-class accuracy of the training trips
and `exercises/data/ex10_svm_seeds.json` (the repeated comparison).

    uv run python exercises/data/make_ex10_svm_map.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "exercises" / "data" / "ex06_ebikes.csv"
OUT_MAP = ROOT / "exercises" / "data" / "ex10_svm_map.npz"
OUT_SEEDS = ROOT / "exercises" / "data" / "ex10_svm_seeds.json"

NUMERIC = ["capacity_wh", "charge_pct", "battery_age_years", "distance_km",
           "ascent_m", "temperature_c", "headwind_kmh", "payload_kg"]
CATEGORICAL = ["surface", "bike_type", "assistance"]
C_RANGE = (1e-2, 1e4)
GAMMA_RANGE = (1e-4, 1e2)
N = 29
N_TRIALS = 25
N_SEEDS = 20


def clean_readings(frame):
    result = frame.copy()
    result.loc[~result["charge_pct"].between(0, 100), "charge_pct"] = np.nan
    result.loc[result["distance_km"] <= 0, "distance_km"] = np.nan
    return result


def make_svm(C=1.0, gamma="scale"):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([
        ("clean", FunctionTransformer(clean_readings)),
        ("prep", ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])),
        ("model", SVC(kernel="rbf", C=C, gamma=gamma))])


ebikes = pd.read_csv(CSV)
X = ebikes[NUMERIC + CATEGORICAL]
y = ebikes["runs_out"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def cv_accuracy(C, gamma):
    return float(cross_val_score(make_svm(C, gamma), X_train, y_train, cv=CV).mean())


def study_best(sampler_name, log, seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        C = trial.suggest_float("C", *C_RANGE, log=log)
        gamma = trial.suggest_float("gamma", *GAMMA_RANGE, log=log)
        return cv_accuracy(C, gamma)

    sampler = (optuna.samplers.TPESampler(seed=seed) if sampler_name == "tpe"
               else optuna.samplers.RandomSampler(seed=seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_value


def main() -> None:
    t0 = time.time()
    log_C = np.linspace(np.log10(C_RANGE[0]), np.log10(C_RANGE[1]), N)
    log_g = np.linspace(np.log10(GAMMA_RANGE[0]), np.log10(GAMMA_RANGE[1]), N)
    pairs = [(i, j) for i in range(N) for j in range(N)]
    values = Parallel(n_jobs=-1)(delayed(cv_accuracy)(10 ** log_C[j], 10 ** log_g[i]) for i, j in pairs)
    cv = np.zeros((N, N))
    for (i, j), v in zip(pairs, values):
        cv[i, j] = v
    default_cv = cv_accuracy(1.0, "scale")
    majority = float(max(y_train.mean(), 1 - y_train.mean()))
    np.savez_compressed(OUT_MAP, log10_C=log_C, log10_gamma=log_g, cv=cv,
                        default_cv=default_cv, majority=majority)
    i, j = np.unravel_index(cv.argmax(), cv.shape)
    print(f"map {N}x{N} in {time.time() - t0:.0f} s; best {cv.max():.3f} at C=1e{log_C[j]:.2f}, "
          f"gamma=1e{log_g[i]:.2f}; default {default_cv:.3f}; majority {majority:.3f}")

    t0 = time.time()
    configs = [(s, log) for s in ["random", "tpe"] for log in [True, False]]
    jobs = [(s, log, seed) for s, log in configs for seed in range(N_SEEDS)]
    bests = Parallel(n_jobs=-1)(delayed(study_best)(s, log, seed) for s, log, seed in jobs)
    summary = {"n_trials": N_TRIALS, "n_seeds": N_SEEDS}
    for s, log in configs:
        b = [v for (s2, l2, _), v in zip(jobs, bests) if (s2, l2) == (s, log)]
        key = f"{s}_{'log' if log else 'linear'}"
        summary[key] = {"mean": float(np.mean(b)), "min": float(np.min(b)), "max": float(np.max(b))}
        print(f"{key:14s} best of {N_TRIALS}: mean {np.mean(b):.3f}, range {np.min(b):.3f}–{np.max(b):.3f}")
    OUT_SEEDS.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"seeds in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
