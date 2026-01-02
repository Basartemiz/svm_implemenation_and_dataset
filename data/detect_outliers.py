

#code for detecting outliers using hdbscan and kmeans
import numpy as np
import os
import hdbscan
from sklearn.cluster import KMeans

def detect_outliers(X, y):
    # Remove outliers using HDBSCAN each class separately

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
    X = np.vstack(final_X)
    y = np.hstack(final_y)

    print(f"Shape after HDBSCAN outlier removal: {X.shape}")

    # Further remove outliers within each class using KMeans
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
    X = np.vstack(final_X)
    y = np.hstack(final_y)

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