"""Random forest: OOB score, number of trees and feature importance.

The out-of-bag samples give a test-error estimate for free — no cross-validation
and no held-out split needed. Here we compare it with 5-fold cross-validation
and with an independent test set, and we plot how both settle as trees are added.

    uv run python lectures/L6/code/random_forest_oob.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

TREE_COUNTS = [25, 50, 100, 200, 400]


def main() -> None:
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target,
                                              test_size=0.3, random_state=0,
                                              stratify=data.target)

    forest = RandomForestClassifier(n_estimators=400, oob_score=True,
                                    max_features="sqrt", random_state=0, n_jobs=-1)
    forest.fit(X_tr, y_tr)
    cv = cross_val_score(RandomForestClassifier(n_estimators=400, random_state=0),
                         X_tr, y_tr, cv=5)
    print(f"OOB score            : {forest.oob_score_:.4f}")
    print(f"5-fold CV accuracy   : {cv.mean():.4f} +- {cv.std():.4f}")
    print(f"independent test set : {forest.score(X_te, y_te):.4f}")

    oob, test = [], []
    for count in TREE_COUNTS:
        model = RandomForestClassifier(n_estimators=count, oob_score=True,
                                       random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        oob.append(model.oob_score_)
        test.append(model.score(X_te, y_te))

    order = np.argsort(forest.feature_importances_)[::-1][:10]
    print("\ntop features:")
    for i in order:
        print(f"  {data.feature_names[i]:<24} {forest.feature_importances_[i]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(TREE_COUNTS, oob, "o-", label="OOB score")
    axes[0].plot(TREE_COUNTS, test, "s-", label="test accuracy")
    axes[0].set(xlabel="number of trees", ylabel="accuracy", xscale="log",
                title="OOB estimate vs. independent test set")
    axes[0].legend()
    axes[1].barh([data.feature_names[i] for i in order][::-1],
                 forest.feature_importances_[order][::-1], color="#007f86")
    axes[1].set(xlabel="importance", title="10 most important features")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
