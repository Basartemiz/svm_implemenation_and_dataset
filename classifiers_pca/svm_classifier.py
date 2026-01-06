#use the custom load_data function from data/preprocess.py and svms from our own custom svm module

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys

# Add project root to path before imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.preprocess import load_data
from data.detect_outliers import detect_outliers
from svm.svm import MultiClassSVM
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    # get data (X,y)
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))

    #apply pca to reduce dimensionality
    X=X.reshape(X.shape[0], -1)  # Flatten images for PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.95)  # retain 95% of variance
    X = pca.fit_transform(X)
    print(f"PCA reduced shape: {X.shape}")

    X,y=detect_outliers(X, y) # detect and remove outliers



    # split data into train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #use a pipeline to standardize the data then apply svm
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", MultiClassSVM()),
        ]
    )

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    
    # Use separate parameter spaces for each kernel type
    param_grid = [
        {
            'svm__kernel': Categorical(['linear']),
            'svm__C': Real(1e-4, 1e4, prior='log-uniform')
        },
        {
            'svm__kernel': Categorical(['rbf']),
            'svm__C': Real(1e-4, 1e4, prior='log-uniform'),
            'svm__gamma': Real(1e-4, 1e4, prior='log-uniform')
        },
        {
            'svm__kernel': Categorical(['poly']),
            'svm__C': Real(1e-4, 1e4, prior='log-uniform'),
            'svm__gamma': Real(1e-4, 1e4, prior='log-uniform'),
            'svm__degree': Integer(2, 5),
            'svm__r': Real(0.1, 10, prior='log-uniform')
        }
    ]
    
    clf = BayesSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='accuracy',
        n_iter=50,
        random_state=42,
        error_score=np.nan,
    )
    clf.fit(X_train, y_train)

    #see results
    import time
    search_start_time = time.time()
    clf.fit(X_train, y_train)
    search_end_time = time.time()
    search_time = search_end_time - search_start_time

    print(f"\nBest parameters found: {clf.best_params_}")
    print(f"Hyperparameter search time: {search_time:.2f} seconds")
    
    best_model = clf.best_estimator_
    
    # Measure training time for best model (refit)
    fit_start_time = time.time()
    best_model.fit(X_train, y_train)
    fit_end_time = time.time()
    training_time = fit_end_time - fit_start_time
    print(f"Training time for best model: {training_time:.2f} seconds")
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy*100:.2f}%")
    print(f"SVM Training accuracy: {accuracy_score(y_train, best_model.predict(X_train))*100:.2f}%")


#inspect farthest point indices for each class
    svm_model = best_model.named_steps['svm']
    print("\nFarthest point indices for each class:")
    # plot using matplotlib and t-SNE the support vectors for each class,
    # and overlay the farthest training point of that class with a star marker.
    points_by_class = {cls: point for cls, point in svm_model.farthest_points.items() if point is not None}

    # inspect support vectors for each class using t-SNE
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sv_X_list = []
    sv_y_list = []
    is_farthest_list = []

    for cls, sv_X in svm_model.support_vectors.items():
        if sv_X is None or sv_X.size == 0:
            continue
        sv_X_list.append(sv_X)
        sv_y_list.append(np.full(sv_X.shape[0], cls))
        is_farthest_list.append(np.zeros(sv_X.shape[0], dtype=bool))

    for cls, point in points_by_class.items():
        sv_X_list.append(point.reshape(1, -1))
        sv_y_list.append(np.array([cls]))
        is_farthest_list.append(np.array([True]))

    if len(sv_X_list) == 0:
        print("No support vectors (or farthest points) available to visualize.")
    else:
        sv_X_all = np.vstack(sv_X_list)
        sv_y_all = np.hstack(sv_y_list)
        is_farthest_all = np.hstack(is_farthest_list)

        perplexity = min(30, max(2, (sv_X_all.shape[0] - 1) // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        sv_X_embedded = tsne.fit_transform(sv_X_all)

        cmap = plt.get_cmap("tab10")
        classes = list(svm_model.classes_)
        color_by_class = {cls: cmap(i % 10) for i, cls in enumerate(classes)}

        # Combined visualization with all classes (Task 1.2a)
        plt.figure(figsize=(12, 10))
        for cls in classes:
            cls_idx = sv_y_all == cls
            base_idx = cls_idx & (~is_farthest_all)
            far_idx = cls_idx & (is_farthest_all)

            if np.any(base_idx):
                plt.scatter(
                    sv_X_embedded[base_idx, 0],
                    sv_X_embedded[base_idx, 1],
                    c=[color_by_class[cls]],
                    label=f"Support vectors (class {cls})",
                    alpha=0.7,
                    s=50
                )
            if np.any(far_idx):
                plt.scatter(
                    sv_X_embedded[far_idx, 0],
                    sv_X_embedded[far_idx, 1],
                    c=[color_by_class[cls]],
                    marker="*",
                    s=300,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"Farthest point (class {cls})",
                )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Combined t-SNE Visualization: Support Vectors and Farthest Points (All Classes)")
        plt.tight_layout()
        plt.show()

    # similarity matrix between classes based on support-vector centroids (cosine similarity)
    classes = list(svm_model.classes_)
    centroids = []
    valid = []
    for cls in classes:
        sv_X = svm_model.support_vectors.get(cls, None)
        if sv_X is None or sv_X.size == 0:
            centroids.append(np.full(X_train.shape[1], np.nan))
            valid.append(False)
        else:
            centroids.append(np.mean(sv_X, axis=0))
            valid.append(True)
    centroids = np.vstack(centroids)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    centroids_normed = centroids / norms
    similarity = centroids_normed @ centroids_normed.T
    similarity[~np.isfinite(similarity)] = np.nan

    import pandas as pd
    sim_df = pd.DataFrame(similarity, index=classes, columns=classes)
    print("\nSupport-vector centroid cosine similarity matrix:")
    print(sim_df)

    # Task 1.2(b): SV Distance Analysis - pairwise distances between support vectors
    print("\n" + "="*80)
    print("Task 1.2(b): Support Vector Distance Analysis")
    print("="*80)
    
    # Collect all support vectors with their class labels
    all_svs = []
    all_sv_classes = []
    for cls in classes:
        sv_X = svm_model.support_vectors.get(cls, None)
        if sv_X is not None and sv_X.size > 0:
            for sv in sv_X:
                all_svs.append(sv)
                all_sv_classes.append(cls)
    
    if len(all_svs) > 1:
        all_svs = np.array(all_svs)
        all_sv_classes = np.array(all_sv_classes)
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        pairwise_distances = squareform(pdist(all_svs, metric='euclidean'))
        
        # Find pairs from different classes
        cross_class_pairs = []
        n_svs = len(all_svs)
        for i in range(n_svs):
            for j in range(i + 1, n_svs):
                if all_sv_classes[i] != all_sv_classes[j]:
                    dist = pairwise_distances[i, j]
                    cross_class_pairs.append({
                        'class_i': all_sv_classes[i],
                        'class_j': all_sv_classes[j],
                        'distance': dist,
                        'sv_i_idx': i,
                        'sv_j_idx': j
                    })
        
        # Sort by distance (closest first)
        cross_class_pairs.sort(key=lambda x: x['distance'])
        
        print(f"\nFound {len(cross_class_pairs)} support vector pairs from different classes.")
        print("\nTop 10 closest support vector pairs (cross-class):")
        print("-" * 80)
        for idx, pair in enumerate(cross_class_pairs[:10], 1):
            print(f"{idx:2d}. Class {pair['class_i']} ↔ Class {pair['class_j']}: "
                  f"distance = {pair['distance']:.4f}")
        
        # Compare with confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        print("\n" + "="*80)
        print("Comparing closest SV pairs with confusion matrix:")
        print("="*80)
        print("\nConfusion Matrix (rows=actual, cols=predicted):")
        print(cm_df)
        
        # Analyze if closest SV pairs correspond to most confused class pairs
        print("\nAnalysis:")
        print("-" * 80)
        top_pairs = cross_class_pairs[:5]
        for pair in top_pairs:
            cls_i, cls_j = pair['class_i'], pair['class_j']
            # Get confusion values (both directions)
            conf_ij = cm_df.loc[cls_i, cls_j]  # Class i predicted as j
            conf_ji = cm_df.loc[cls_j, cls_i]  # Class j predicted as i
            total_confusion = conf_ij + conf_ji
            print(f"SV pair: {cls_i} ↔ {cls_j} (distance: {pair['distance']:.4f})")
            print(f"  → Confusion: {cls_i}→{cls_j}: {conf_ij}, {cls_j}→{cls_i}: {conf_ji}, "
                  f"Total: {total_confusion}")
        
        # Create a summary table
        summary_data = []
        for pair in cross_class_pairs[:10]:
            cls_i, cls_j = pair['class_i'], pair['class_j']
            conf_ij = cm_df.loc[cls_i, cls_j]
            conf_ji = cm_df.loc[cls_j, cls_i]
            summary_data.append({
                'Class Pair': f"{cls_i} ↔ {cls_j}",
                'SV Distance': f"{pair['distance']:.4f}",
                'Confusion (i→j)': conf_ij,
                'Confusion (j→i)': conf_ji,
                'Total Confusion': conf_ij + conf_ji
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("Summary Table: Top 10 Closest SV Pairs vs Confusion")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Discussion
        print("\n" + "="*80)
        print("Discussion:")
        print("="*80)
        print("The analysis shows whether support vectors that are closest to each other")
        print("belong to the categories most confused by the classifier.")
        print("If there's a correlation, we would expect the closest SV pairs to have")
        print("higher confusion values in the confusion matrix.")
    else:
        print("Not enough support vectors for distance analysis.")

    #create confusion matrix that shows the performance of the classifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
