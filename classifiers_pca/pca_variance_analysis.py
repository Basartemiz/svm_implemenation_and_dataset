"""
PCA Variance Analysis: How variance threshold affects accuracy and training time

This script tests different PCA variance thresholds and compares:
- Number of components
- Training time
- Classification accuracy
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.preprocess import load_data


def test_classifier_with_pca(X_train, X_test, y_train, y_test, variance_threshold):
    """Test classifier with given PCA variance threshold."""
    # Apply PCA if threshold is provided
    if variance_threshold is None:
        # No PCA - use original features
        X_train_pca = X_train
        X_test_pca = X_test
        n_components = X_train.shape[1]
        variance_retained = 1.0
    else:
        pca = PCA(n_components=variance_threshold)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        n_components = X_train_pca.shape[1]
        variance_retained = np.sum(pca.explained_variance_ratio_)
    
    # Train classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
    
    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'variance_threshold': variance_threshold if variance_threshold is not None else 1.0,
        'n_components': n_components,
        'variance_retained': variance_retained,
        'training_time': training_time,
        'accuracy': accuracy
    }


def main():
    """Main function to analyze PCA variance effects."""
    print("="*80)
    print("PCA Variance Analysis: Accuracy vs Speed Trade-off")
    print("="*80)
    
    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))
    
    # Flatten images
    X = X.reshape(X.shape[0], -1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nOriginal feature dimension: {X_train_scaled.shape[1]}")
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    
    # Test different variance thresholds
    variance_thresholds = [0.80, 0.85, 0.90, 0.95, 0.99, 1.0]  # 1.0 = no PCA
    results = []
    
    print("\nTesting different variance thresholds...")
    print("-" * 80)
    
    for var_thresh in variance_thresholds:
        if var_thresh == 1.0:
            # No PCA - use original features
            result = test_classifier_with_pca(
                X_train_scaled, X_test_scaled, y_train, y_test, None
            )
        else:
            result = test_classifier_with_pca(
                X_train_scaled, X_test_scaled, y_train, y_test, var_thresh
            )
        
        results.append(result)
        print(f"Variance {result['variance_threshold']:.2f}: {result['n_components']:4d} components | "
              f"Accuracy: {result['accuracy']:.4f} | Time: {result['training_time']:.4f}s")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy vs Number of Components
    axes[0].plot(df['n_components'], df['accuracy'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy vs Number of Components')
    axes[0].grid(True, alpha=0.3)
    for i, row in df.iterrows():
        axes[0].annotate(f"{row['variance_threshold']:.2f}", 
                        (row['n_components'], row['accuracy']),
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot 2: Training Time vs Number of Components
    axes[1].plot(df['n_components'], df['training_time'], 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time vs Number of Components')
    axes[1].grid(True, alpha=0.3)
    for i, row in df.iterrows():
        axes[1].annotate(f"{row['variance_threshold']:.2f}", 
                        (row['n_components'], row['training_time']),
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot 3: Accuracy vs Training Time (trade-off)
    axes[2].scatter(df['training_time'], df['accuracy'], s=100, alpha=0.7, c=df['n_components'], cmap='viridis')
    axes[2].set_xlabel('Training Time (seconds)')
    axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('Accuracy vs Training Time Trade-off')
    axes[2].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label('Number of Components')
    for i, row in df.iterrows():
        axes[2].annotate(f"{row['variance_threshold']:.2f}", 
                        (row['training_time'], row['accuracy']),
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # Save results
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'pca_variance_analysis.png'), dpi=150, bbox_inches='tight')
    df.to_csv(os.path.join(results_dir, 'pca_variance_analysis_results.csv'), index=False)
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nPlots saved to results/pca_variance_analysis.png")
    print(f"Results saved to results/pca_variance_analysis_results.csv")
    
    # Analysis
    print("\n" + "="*80)
    print("Analysis:")
    print("="*80)
    print("\nKey Findings:")
    print(f"1. Higher variance threshold (e.g., 0.95) = MORE components = SLOWER but HIGHER accuracy")
    print(f"2. Lower variance threshold (e.g., 0.80) = FEWER components = FASTER but LOWER accuracy")
    print(f"3. Optimal trade-off: Choose variance threshold based on your accuracy/speed requirements")
    
    best_accuracy_idx = df['accuracy'].idxmax()
    fastest_idx = df['training_time'].idxmin()
    best_tradeoff_idx = (df['accuracy'] / df['training_time']).idxmax()
    
    print(f"\nBest Accuracy: {df.loc[best_accuracy_idx, 'variance_threshold']:.2f} variance "
          f"({df.loc[best_accuracy_idx, 'n_components']} components, "
          f"accuracy={df.loc[best_accuracy_idx, 'accuracy']:.4f})")
    print(f"Fastest Training: {df.loc[fastest_idx, 'variance_threshold']:.2f} variance "
          f"({df.loc[fastest_idx, 'n_components']} components, "
          f"time={df.loc[fastest_idx, 'training_time']:.4f}s)")
    print(f"Best Trade-off: {df.loc[best_tradeoff_idx, 'variance_threshold']:.2f} variance "
          f"({df.loc[best_tradeoff_idx, 'n_components']} components)")
    
    plt.show()
    
    return df


if __name__ == "__main__":
    main()

