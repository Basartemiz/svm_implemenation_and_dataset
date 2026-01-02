#we shall make outlier detection function here using hdbscan and kmeans


from data.preprocess import load_data
from data.detect_outliers import detect_outliers
import numpy as np
import os


def main():
    # get data (X,y)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))


    #use pca to reduce dimensionality
    X=X.reshape(X.shape[0], -1)  # Flatten images for PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)  # retain 95% of variance
    X = pca.fit_transform(X)
    print(f"PCA reduced shape: {X.shape}")
    #visualize original data shape using t-sne
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE


    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(8,6))
    tsne_results = tsne.fit_transform(X.reshape(X.shape[0], -1))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y)
    plt.show()


    X, y = detect_outliers(X, y)  # detect and remove outliers
    #use pca to reduce dimensionality
    X=X.reshape(X.shape[0], -1)  # Flatten images for PCA
    pca = PCA(n_components=0.95)  # retain 95% of variance
    X = pca.fit_transform(X)
    print(f"PCA reduced shape after outlier removal: {X.shape}")
    #visualize data shape
    #after the outlier removal
    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(8,6))
    tsne_results = tsne.fit_transform(X.reshape(X.shape[0], -1))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y)
    plt.show()

    return X, y

if __name__ == "__main__":
    main()