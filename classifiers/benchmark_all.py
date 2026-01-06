import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def _ensure_project_root_on_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def _stratified_subset(X, y, max_per_class, random_state=42):
    if max_per_class is None:
        return X, y
    rng = np.random.default_rng(random_state)
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = min(int(max_per_class), int(cls_idx.size))
        indices.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    indices = np.array(indices, dtype=int)
    return X[indices], y[indices]


def _safe_train_test_split(X, y, test_size, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = int(y.shape[0])
    n_classes = int(classes.size)

    test_n = int(round(n_samples * float(test_size))) if isinstance(test_size, float) else int(test_size)
    test_n = max(test_n, n_classes)
    test_n = min(test_n, n_samples - 1) if n_samples > 1 else 0

    do_stratify = bool(stratify) and (counts.min() >= 2) and (test_n >= n_classes)
    try:
        return train_test_split(
            X,
            y,
            test_size=test_n,
            random_state=random_state,
            stratify=y if do_stratify else None,
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_n, random_state=random_state, stratify=None)


def _benchmark_one(name, estimator, params, X_train, y_train, X_test, y_test):
    try:
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        y_pred = estimator.predict(X_test)
        acc = float(np.mean(y_pred == y_test))
        return {
            "classifier": name,
            "status": "ok",
            "train_time_sec": float(train_time),
            "test_accuracy": acc,
            "params": json.dumps(params, default=str),
            "error": "",
        }
    except Exception as e:
        return {
            "classifier": name,
            "status": "failed",
            "train_time_sec": np.nan,
            "test_accuracy": np.nan,
            "params": json.dumps(params, default=str),
            "error": f"{type(e).__name__}: {e}",
        }


def _skipped(name, params, reason, X_train, X_test):
    return {
        "classifier": name,
        "status": "skipped",
        "train_time_sec": np.nan,
        "test_accuracy": np.nan,
        "params": json.dumps(params, default=str),
        "error": reason,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }


def main():
    project_root = _ensure_project_root_on_path()

    parser = argparse.ArgumentParser(description="Benchmark all classifiers (no PCA) and report training times.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for samples per class.")
    args = parser.parse_args()

    from data.preprocess import load_data
    from data.detect_outliers import detect_outliers_svm_slack

    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))
    X, y = _stratified_subset(X, y, args.max_per_class)

    X, y = detect_outliers_svm_slack(
        X,
        y,
        C=1.0,
        kernel="rbf",
        gamma=0.1,
        outlier_threshold_percentile=95,
        verbose=False,
    )

    X_train, X_test, y_train, y_test = _safe_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import PolynomialFeatures

    from svm.svm import MultiClassSVM

    chosen = {
        "logistic_regression": {"C": 0.000774263682681127, "solver": "lbfgs", "max_iter": 1000},
        "knn": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
        "naive_bayes": {"var_smoothing": 1e-9},
        "random_forest": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": 1,
        },
        "svm": {"C": 50.00005, "kernel": "rbf", "gamma": 0.0001, "degree": 2, "r": 0.1},
        # This is extremely expensive on 12,288-D raw pixels; keep it small and mark failures if any.
        "logistic_regression_nonlinear": {"poly_degree": 2, "C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    }

    models = [
        (
            "LogReg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "log_reg",
                        LogisticRegression(
                            C=chosen["logistic_regression"]["C"],
                            solver=chosen["logistic_regression"]["solver"],
                            max_iter=chosen["logistic_regression"]["max_iter"],
                        ),
                    ),
                ]
            ),
            chosen["logistic_regression"],
        ),
        (
            "kNN",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "knn",
                        KNeighborsClassifier(
                            n_neighbors=chosen["knn"]["n_neighbors"],
                            weights=chosen["knn"]["weights"],
                            metric=chosen["knn"]["metric"],
                        ),
                    ),
                ]
            ),
            chosen["knn"],
        ),
        (
            "NaiveBayes",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("nb", GaussianNB(var_smoothing=chosen["naive_bayes"]["var_smoothing"])),
                ]
            ),
            chosen["naive_bayes"],
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=chosen["random_forest"]["n_estimators"],
                            max_depth=chosen["random_forest"]["max_depth"],
                            min_samples_split=chosen["random_forest"]["min_samples_split"],
                            min_samples_leaf=chosen["random_forest"]["min_samples_leaf"],
                            random_state=chosen["random_forest"]["random_state"],
                            n_jobs=chosen["random_forest"]["n_jobs"],
                        ),
                    ),
                ]
            ),
            chosen["random_forest"],
        ),
        (
            "SVM (scratch)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svm",
                        MultiClassSVM(
                            C=chosen["svm"]["C"],
                            kernel=chosen["svm"]["kernel"],
                            gamma=chosen["svm"]["gamma"],
                            degree=chosen["svm"]["degree"],
                            r=chosen["svm"]["r"],
                        ),
                    ),
                ]
            ),
            chosen["svm"],
        ),
        (
            "LogReg (Poly)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=chosen["logistic_regression_nonlinear"]["poly_degree"])),
                    (
                        "log_reg",
                        LogisticRegression(
                            C=chosen["logistic_regression_nonlinear"]["C"],
                            solver=chosen["logistic_regression_nonlinear"]["solver"],
                            max_iter=chosen["logistic_regression_nonlinear"]["max_iter"],
                        ),
                    ),
                ]
            ),
            chosen["logistic_regression_nonlinear"],
        ),
    ]

    rows = []
    for name, estimator, params in models:
        if name == "LogReg (Poly)" and X_train.shape[1] > 500:
            rows.append(
                _skipped(
                    name,
                    params,
                    "PolynomialFeatures on raw pixels is infeasible; run the PCA benchmark instead.",
                    X_train,
                    X_test,
                )
            )
            continue
        row = _benchmark_one(name, estimator, params, X_train, y_train, X_test, y_test)
        row["n_train"] = int(X_train.shape[0])
        row["n_test"] = int(X_test.shape[0])
        row["n_features"] = int(X_train.shape[1])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["status", "train_time_sec"], ascending=[True, True], na_position="last")

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "benchmark_training_times.csv")
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
