#cluster the data after reducing dimensions with PCA

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
    
    # Task 2.3: Compare clustering-based outlier detection with SVM-based outlier detection
    print("\n" + "="*80)
    print("Task 2.3: Clustering-based vs SVM-based Outlier Detection Comparison")
    print("="*80)
    
    # Clustering-based outlier detection (using HDBSCAN as in detect_outliers.py)
    from data.detect_outliers import detect_outliers
    X_clustering_clean, y_clustering_clean = detect_outliers(X_pca, y)
    
    print(f"\nClustering-based (HDBSCAN + KMeans) outlier removal:")
    print(f"  Original samples: {len(X_pca)}")
    print(f"  After removal: {len(X_clustering_clean)}")
    print(f"  Removed: {len(X_pca) - len(X_clustering_clean)} "
          f"({100*(len(X_pca) - len(X_clustering_clean))/len(X_pca):.2f}%)")
    
    # SVM-based outlier detection
    try:
        sys.path.insert(0, project_root)
        from classifiers_pca.svm_outlier_detection import detect_outliers_svm
        
        X_svm_clean, y_svm_clean, svm_outlier_indices, _ = detect_outliers_svm(
            X_pca, y, C=1.0, kernel='rbf', gamma=0.1, outlier_threshold_percentile=95
        )
        
        print(f"\nSVM-based outlier removal:")
        print(f"  Original samples: {len(X_pca)}")
        print(f"  After removal: {len(X_svm_clean)}")
        print(f"  Removed: {len(X_pca) - len(X_svm_clean)} "
              f"({100*(len(X_pca) - len(X_svm_clean))/len(X_pca):.2f}%)")
        
        # Comparison summary
        print("\n" + "="*80)
        print("Comparison Summary")
        print("="*80)
        print(f"Clustering method removed: {len(X_pca) - len(X_clustering_clean)} samples")
        print(f"SVM method removed: {len(X_pca) - len(X_svm_clean)} samples")
        print(f"Difference: {abs(len(X_clustering_clean) - len(X_svm_clean))} samples")
        
        # Save comparison results
        import pandas as pd
        comparison_data = {
            'method': ['Clustering (HDBSCAN+KMeans)', 'SVM (slack variables)'],
            'samples_after_removal': [len(X_clustering_clean), len(X_svm_clean)],
            'samples_removed': [len(X_pca) - len(X_clustering_clean), 
                              len(X_pca) - len(X_svm_clean)],
            'removal_percentage': [100*(len(X_pca) - len(X_clustering_clean))/len(X_pca),
                                  100*(len(X_pca) - len(X_svm_clean))/len(X_pca)]
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(results_dir, 'outlier_detection_comparison.csv'), index=False)
        print(f"\nComparison results saved to results/outlier_detection_comparison.csv")
        
    except Exception as e:
        print(f"\nNote: SVM outlier detection comparison skipped: {e}")
        print("Run classifiers_pca/svm_outlier_detection.py separately to generate comparison.")

if __name__ == "__main__":
    import sys
    main()