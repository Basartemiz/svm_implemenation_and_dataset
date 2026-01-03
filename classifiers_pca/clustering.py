#cluster the data after reducing dimensions with PCA


import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from  data.preprocess import load_data


def main():
    # get data (X,y)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))

    # Flatten images if necessary
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Reduce dimensions with PCA
    pca = PCA(n_components=50)  # reduce to 50 dimensions
    X_pca = pca.fit_transform(X_flat)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
    y_pred = kmeans.fit_predict(X_pca)

    # Visualize the clusters in 2D PCA space
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_flat)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.title('KMeans Clustering after PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()


    #evaluate clustering performance using silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X_pca, y_pred)
    print("Silhouette Score: ", silhouette_avg)

    #evaluate clustering performance using adjusted rand index
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y, y_pred)
    print("Adjusted Rand Index: ", ari)

if __name__ == "__main__":
    main()