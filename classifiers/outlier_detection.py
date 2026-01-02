#we must detect outlier before splitting the data to avoid data leakage



from data.preprocess import load_data
import numpy as np
from data.detect_outliers import detect_outliers_isolation_forest


def main():
    # get data (X,y)
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))
    #visualize original data shape using t-sne
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(8,6))
    tsne_results = tsne.fit_transform(X.reshape(X.shape[0], -1))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y, cmap='viridis', alpha=0.7)
    plt.title("t-SNE of Original Fruit Images Dataset")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label='Class Label')
    plt.show()


    X,y=detect_outliers_isolation_forest(X, y) # detect and remove outliers

    #visualize data shape after outlier removal using t-sne
    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(8,6))
    tsne_results = tsne.fit_transform(X.reshape(X.shape[0], -1))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y, cmap='viridis', alpha=0.7)
    plt.title("t-SNE of Fruit Images Dataset After Outlier Removal")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label='Class Label')
    plt.show()

    return X, y
if __name__ == "__main__":
    main()