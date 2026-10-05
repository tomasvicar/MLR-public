"""SVM on real data: standardization, tuning C and gamma, testing.

Puts the whole workflow from the previous lectures together — a pipeline with
standardization inside the cross-validation, a grid search over the
hyperparameters, and only at the very end an evaluation on a held-out test set.

    uv run python lectures/L4/code/svm_applications.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0)

    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10, 100]},
        {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10, 100],
         "svm__gamma": [0.001, 0.01, 0.1, 1]},
        {"svm__kernel": ["poly"], "svm__C": [0.1, 1, 10],
         "svm__degree": [2, 3, 5]},
    ]

    search = GridSearchCV(pipeline, grid, cv=5, scoring="balanced_accuracy",
                          n_jobs=-1)
    search.fit(X_train, y_train)

    print("best parameters:", search.best_params_)
    print("CV balanced accuracy:", round(search.best_score_, 4))
    print("support vectors:", search.best_estimator_["svm"].n_support_,
          "of", len(y_train), "training samples")
    print()
    print(classification_report(y_test, search.predict(X_test),
                                target_names=["malignant", "benign"]))


if __name__ == "__main__":
    main()
