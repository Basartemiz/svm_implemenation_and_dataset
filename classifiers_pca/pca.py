"""
PCA Exploration (Task 2.1)

This module explores the intrinsic dimensionality of features using:
1. Cumulative explained variance ratio
2. Reconstruction error vs. number of components
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from data.preprocess import load_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_reconstruction_error(X, n_components):
    """Compute reconstruction error for given number of components."""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    error = np.mean((X - X_reconstructed) ** 2)
    return error


def main():
    """Main function for PCA exploration."""
    print("="*80)
    print("PCA Exploration (Task 2.1)")
    print("="*80)
    
    # get data (X,y)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))

    # Flatten images if necessary
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    
    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)

    print(f"Original shape: {X_flat_scaled.shape}")
    
    # Fit PCA with all components to get variance information
    pca_full = PCA()
    pca_full.fit(X_flat_scaled)
    
    # 1. Plot cumulative explained variance ratio
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
    plt.axhline(y=0.80, color='g', linestyle='--', label='80% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Find number of components for different variance thresholds
    n_comp_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_comp_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_comp_80 = np.argmax(cumulative_variance >= 0.80) + 1
    
    print(f"\nNumber of components for variance retention:")
    print(f"  95% variance: {n_comp_95} components")
    print(f"  90% variance: {n_comp_90} components")
    print(f"  80% variance: {n_comp_80} components")
    
    # 2. Plot reconstruction error vs. number of components
    n_components_range = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    n_components_range = [n for n in n_components_range if n <= X_flat_scaled.shape[1]]
    reconstruction_errors = []
    
    print("\nComputing reconstruction errors...")
    for n_comp in n_components_range:
        error = compute_reconstruction_error(X_flat_scaled, n_comp)
        reconstruction_errors.append(error)
        print(f"  {n_comp} components: reconstruction error = {error:.4f}")
    
    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, reconstruction_errors, 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Reconstruction Error')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'pca_exploration.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to results/pca_exploration.png")
    plt.show()
    
    # 3. Apply PCA with 95% variance (as used in classifiers)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_flat_scaled)
    
    print(f"\nPCA reduced shape (95% variance): {X_pca.shape}")
    print(f"Variance retained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Save PCA results
    pca_results = {
        'n_components_95': n_comp_95,
        'n_components_90': n_comp_90,
        'n_components_80': n_comp_80,
        'variance_95': cumulative_variance[n_comp_95 - 1] if n_comp_95 <= len(cumulative_variance) else 1.0,
        'variance_90': cumulative_variance[n_comp_90 - 1] if n_comp_90 <= len(cumulative_variance) else 1.0,
        'variance_80': cumulative_variance[n_comp_80 - 1] if n_comp_80 <= len(cumulative_variance) else 1.0,
    }
    
    import pandas as pd
    pca_df = pd.DataFrame([pca_results])
    pca_df.to_csv(os.path.join(results_dir, 'pca_exploration_results.csv'), index=False)
    print(f"PCA results saved to results/pca_exploration_results.csv")

    return X_pca, y


if __name__ == "__main__":
    main()