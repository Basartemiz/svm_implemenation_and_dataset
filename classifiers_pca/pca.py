#use pca to eliminate redundant features and speed up training

from data.preprocess import load_data
from sklearn.decomposition import PCA
import numpy as np



def main():
    # get data (X,y)
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))

    # Flatten images if necessary
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Apply PCA
    pca = PCA(n_components=0.95)  # retain 95% of variance
    X_pca = pca.fit_transform(X_flat)

    print(f"Original shape: {X_flat.shape}, PCA reduced shape: {X_pca.shape}")

    return X_pca, y
if __name__ == "__main__":
    main()