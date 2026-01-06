

#code for detecting outliers using hdbscan and kmeans
import numpy as np
import os
import hdbscan
from sklearn.cluster import KMeans

def detect_outliers(X, y):
    # Remove outliers using HDBSCAN each class separately

    X_original = X
    y_original = y

    final_X = []
    final_y = []
    for cls in np.unique(y):
        cls_mask = y == cls
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]

        if X_cls.shape[0] < 2:
            continue  # skip classes with less than 2 samples

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(X_cls)

        # Keep only points that are not labeled as -1 (outliers)
        inlier_mask = cluster_labels != -1

        final_X.append(X_cls[inlier_mask])
        final_y.append(y_cls[inlier_mask])
    if len(final_X) == 0:
        print("HDBSCAN outlier removal produced no inliers; skipping this step.")
        return X, y
    X = np.vstack(final_X)
    y = np.hstack(final_y)
    if X.shape[0] == 0:
        print("HDBSCAN outlier removal produced zero inliers; skipping this step.")
        return X_original, y_original

    print(f"Shape after HDBSCAN outlier removal: {X.shape}")

    # Further remove outliers within each class using KMeans
    X_after_hdbscan = X
    y_after_hdbscan = y
    final_X = []
    final_y = []
    for cls in np.unique(y):
        cls_mask = y == cls
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]

        if X_cls.shape[0] < 2:
            continue  # skip classes with less than 2 samples

        kmeans = KMeans(n_clusters=1, random_state=42) # single cluster to find center
        kmeans.fit(X_cls)
        center = kmeans.cluster_centers_[0]
        distances = np.linalg.norm(X_cls - center, axis=1)
        threshold = np.percentile(distances, 90)  # keep 90% closest points
        inlier_mask = distances <= threshold

        final_X.append(X_cls[inlier_mask])
        final_y.append(y_cls[inlier_mask])
    if len(final_X) == 0:
        print("KMeans outlier removal produced no inliers; skipping this step.")
        return X, y
    X = np.vstack(final_X)
    y = np.hstack(final_y)
    if X.shape[0] == 0:
        print("KMeans outlier removal produced zero inliers; skipping this step.")
        return X_after_hdbscan, y_after_hdbscan

    print(f"Shape after KMeans outlier removal: {X.shape}")
    return X, y


def detect_outliers_isolation_forest(X, y):
    from sklearn.ensemble import IsolationForest

    iso_forest = IsolationForest(contamination=0.4, random_state=42)
    #run isolation forest on each class separately
    final_X = []
    final_y = []
    for cls in np.unique(y):
        cls_mask = y == cls
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]

        iso_forest.fit(X_cls)
        preds = iso_forest.predict(X_cls)
        inlier_mask = preds == 1  # keep only inliers

        final_X.append(X_cls[inlier_mask])
        final_y.append(y_cls[inlier_mask])
    X = np.vstack(final_X)
    y = np.hstack(final_y)

    print(f"Shape after Isolation Forest outlier removal: {X.shape}")
    return X, y


def detect_outliers_svm_slack(
    X,
    y,
    *,
    C=1.0,
    kernel="rbf",
    gamma=0.1,
    degree=3,
    r=1.0,
    outlier_threshold_percentile=95,
    verbose=True,
):
    """
    Detect outliers via soft-margin SVM slack variables.

    For each one-vs-rest classifier, compute slack:
      ξᵢ = max(0, 1 - yᵢ f(xᵢ))
    and mark points above a per-class percentile threshold as outliers.
    """
    from sklearn.preprocessing import StandardScaler
    from svm.svm import MultiClassSVM

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have compatible shapes")
    if X.shape[0] < 2:
        return X, y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm_model = MultiClassSVM(C=C, kernel=kernel, gamma=gamma, degree=degree, r=r)
    svm_model.fit(X_scaled, y)

    all_outliers = np.zeros(len(y), dtype=bool)
    for cls in svm_model.classes_:
        y_binary = np.where(y == cls, 1, -1)
        decision_values = svm_model.models[cls].predict(X_scaled).reshape(-1)
        slack = np.maximum(0, 1 - y_binary * decision_values)
        threshold = np.percentile(slack, outlier_threshold_percentile)
        flags = slack > threshold
        all_outliers |= flags
        if verbose:
            print(
                f"Class {cls}: {np.sum(flags)} outliers detected "
                f"(threshold: {threshold:.4f}, max slack: {np.max(slack):.4f})"
            )

    inliers = ~all_outliers
    X_clean = X[inliers]
    y_clean = y[inliers]
    print(f"Shape after SVM-slack outlier removal: {X_clean.shape}")
    return X_clean, y_clean
