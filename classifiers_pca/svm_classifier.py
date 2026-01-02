#use the custom load_data function from data/preprocess.py and svms from our own custom svm module


from data.preprocess import load_data
from data.detect_outliers import detect_outliers
from svm.svm import MultiClassSVM
import os
import sys
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
        X, y, test_size=0.4, random_state=42, stratify=y
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

    parameters={
        'svm__C':np.logspace(-4,4,10),
        'svm__kernel': ['linear', 'rbf', 'poly'], 
        'svm__degree': [2,3,4,5] , # only used for 'poly' kernel
        'svm__gamma': np.logspace(-4, 4, 10),  # only used for 'rbf' and 'poly' kernels
        'svm__r':[0.1, 1,10]  # only used for 'rbf' and 'poly' kernels
    }

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, scoring='accuracy')
    clf.fit(X_train, y_train)

    #see results

    print("Best parameters found: ", clf.best_params_)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy*100:.2f}%")

    #get training accuracy using best estimator
    import time
    start_time = time.time()
    best_model = clf.best_estimator_
    best_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time for best model: {training_time:.2f} seconds")
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

        for cls in classes:
            cls_idx = sv_y_all == cls
            base_idx = cls_idx & (~is_farthest_all)
            far_idx = cls_idx & (is_farthest_all)

            plt.figure(figsize=(10, 8))
            if np.any(base_idx):
                plt.scatter(
                    sv_X_embedded[base_idx, 0],
                    sv_X_embedded[base_idx, 1],
                    c=[color_by_class[cls]],
                    label=f"Support vectors (class {cls})",
                )
            if np.any(far_idx):
                plt.scatter(
                    sv_X_embedded[far_idx, 0],
                    sv_X_embedded[far_idx, 1],
                    c=[color_by_class[cls]],
                    marker="*",
                    s=300,
                    edgecolors="black",
                    linewidths=1.0,
                    label=f"Farthest point (class {cls})",
                )
            plt.legend()
            plt.title(f"t-SNE visualization (class {cls})")
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

    #create confusion matrix that shows the performance of the classifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
