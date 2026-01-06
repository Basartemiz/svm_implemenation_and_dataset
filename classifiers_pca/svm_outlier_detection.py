"""
SVM-based Outlier Detection (Task 1.4)

This module implements outlier detection using SVM constraints (slack variables).
The approach uses soft-margin SVM and identifies outliers as points with high slack values.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.preprocess import load_data
from svm.svm import MultiClassSVM


def compute_slack_variables(X, y, svm_model, outlier_threshold_percentile=95):
    """
    Compute slack variables for each point using SVM constraints.
    
    Slack variable: ξᵢ = max(0, 1 - yᵢ(w·xᵢ + b))
    Points with ξᵢ > threshold are considered outliers (margin violators).
    
    Args:
        X: Feature matrix
        y: Labels
        svm_model: Trained MultiClassSVM model
    
    Returns:
        slack_values: Dictionary mapping class to slack values
        outlier_flags: Dictionary mapping class to boolean outlier flags
    """
    slack_values = {}
    outlier_flags = {}
    
    # X should already be scaled (from the training pipeline)
    X_scaled = X
    
    for cls in svm_model.classes_:
        # Get binary labels for this class
        y_binary = np.where(y == cls, 1, -1)
        
        # Get the corresponding SVM model
        svm_binary = svm_model.models[cls]
        
        # Compute decision function values f(x) using the model's own predict()
        # (returns raw decision values, not class labels).
        decision_values = svm_binary.predict(X_scaled).reshape(-1)
        
        # Compute slack: ξ = max(0, 1 - y * f(x))
        slack = np.maximum(0, 1 - y_binary * decision_values)
        slack_values[cls] = slack
        
        # Identify outliers: points with slack > threshold
        threshold = np.percentile(slack, outlier_threshold_percentile)
        outlier_flags[cls] = slack > threshold
        
        print(f"Class {cls}: {np.sum(outlier_flags[cls])} outliers detected "
              f"(threshold: {threshold:.4f}, max slack: {np.max(slack):.4f})")
    
    return slack_values, outlier_flags


def detect_outliers_svm(X, y, C=1.0, kernel='rbf', gamma=0.1, outlier_threshold_percentile=95):
    """
    Detect outliers using SVM slack variables.
    
    Args:
        X: Feature matrix
        y: Labels
        C: SVM regularization parameter
        kernel: Kernel type ('linear', 'rbf', 'poly')
        gamma: RBF/poly kernel parameter
        outlier_threshold_percentile: Percentile to use as outlier threshold
    
    Returns:
        X_clean: Feature matrix with outliers removed
        y_clean: Labels with outliers removed
        outlier_indices: Indices of detected outliers
        slack_values: Slack values for all points
    """
    print("="*80)
    print("SVM-based Outlier Detection (Task 1.4)")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print(f"\nTraining SVM (C={C}, kernel={kernel})...")
    svm_model = MultiClassSVM(C=C, kernel=kernel, gamma=gamma)
    svm_model.fit(X_train_scaled, y_train)
    
    # Compute slack variables on training set
    print("\nComputing slack variables...")
    slack_values, outlier_flags = compute_slack_variables(
        X_train_scaled,
        y_train,
        svm_model,
        outlier_threshold_percentile=outlier_threshold_percentile,
    )
    
    # Combine outlier flags across all classes
    all_outlier_flags = np.zeros(len(y_train), dtype=bool)
    for cls in svm_model.classes_:
        all_outlier_flags |= outlier_flags[cls]
    
    # Get outlier indices
    outlier_indices = np.where(all_outlier_flags)[0]
    inlier_indices = np.where(~all_outlier_flags)[0]
    
    print(f"\nTotal outliers detected: {len(outlier_indices)} out of {len(y_train)} "
          f"({100*len(outlier_indices)/len(y_train):.2f}%)")
    
    # Remove outliers
    X_clean = X_train[inlier_indices]
    y_clean = y_train[inlier_indices]
    
    # Visualize outliers
    visualize_outliers(X_train_scaled, y_train, outlier_flags, slack_values, svm_model)
    
    return X_clean, y_clean, outlier_indices, slack_values


def visualize_outliers(X, y, outlier_flags, slack_values, svm_model):
    """Visualize detected outliers using t-SNE."""
    from sklearn.manifold import TSNE
    
    print("\nGenerating visualization...")
    
    # Use t-SNE for 2D visualization
    perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_embedded = tsne.fit_transform(X)
    
    # Create figure with subplots for each class
    classes = list(svm_model.classes_)
    n_classes = len(classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    cmap = plt.get_cmap("tab10")
    
    for idx, cls in enumerate(classes):
        ax = axes[idx]
        cls_mask = y == cls
        inlier_mask = cls_mask & (~outlier_flags[cls])
        outlier_mask = cls_mask & outlier_flags[cls]
        
        # Plot inliers
        if np.any(inlier_mask):
            ax.scatter(
                X_embedded[inlier_mask, 0],
                X_embedded[inlier_mask, 1],
                c=[cmap(idx % 10)],
                alpha=0.5,
                s=30,
                label='Inliers',
                edgecolors='none'
            )
        
        # Plot outliers
        if np.any(outlier_mask):
            ax.scatter(
                X_embedded[outlier_mask, 0],
                X_embedded[outlier_mask, 1],
                c='red',
                marker='x',
                s=100,
                linewidths=2,
                label='Outliers',
                zorder=10
            )
        
        ax.set_title(f'Class {cls} - Outlier Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'results', 'svm_outlier_detection.png'), dpi=150)
    print(f"Visualization saved to results/svm_outlier_detection.png")
    plt.show()
    
    # Combined plot
    plt.figure(figsize=(12, 10))
    for idx, cls in enumerate(classes):
        cls_mask = y == cls
        inlier_mask = cls_mask & (~outlier_flags[cls])
        outlier_mask = cls_mask & outlier_flags[cls]
        
        if np.any(inlier_mask):
            plt.scatter(
                X_embedded[inlier_mask, 0],
                X_embedded[inlier_mask, 1],
                c=[cmap(idx % 10)],
                alpha=0.5,
                s=30,
                label=f'Class {cls} (inliers)',
                edgecolors='none'
            )
        
        if np.any(outlier_mask):
            plt.scatter(
                X_embedded[outlier_mask, 0],
                X_embedded[outlier_mask, 1],
                c='red',
                marker='x',
                s=100,
                linewidths=2,
                label=f'Class {cls} (outliers)',
                zorder=10
            )
    
    plt.title('SVM-based Outlier Detection - All Classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'results', 'svm_outlier_detection_combined.png'), dpi=150)
    plt.show()


def main():
    """Main function for SVM-based outlier detection."""
    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))
    
    # Flatten and apply PCA
    X = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X)
    print(f"PCA reduced shape: {X.shape}")
    
    # Detect outliers using SVM
    X_clean, y_clean, outlier_indices, slack_values = detect_outliers_svm(
        X, y, C=1.0, kernel='rbf', gamma=0.1, outlier_threshold_percentile=95
    )
    
    print("\n" + "="*80)
    print("Outlier Detection Summary")
    print("="*80)
    print(f"Original samples: {len(X)}")
    print(f"After outlier removal: {len(X_clean)}")
    print(f"Outliers removed: {len(X) - len(X_clean)} ({100*(len(X) - len(X_clean))/len(X):.2f}%)")
    
    # Save results
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save slack values
    all_slacks = []
    all_classes = []
    for cls in slack_values.keys():
        all_slacks.extend(slack_values[cls])
        all_classes.extend([cls] * len(slack_values[cls]))
    
    slack_df = pd.DataFrame({
        'class': all_classes,
        'slack_value': all_slacks
    })
    slack_df.to_csv(os.path.join(results_dir, 'svm_outlier_slack_values.csv'), index=False)
    print(f"\nSlack values saved to results/svm_outlier_slack_values.csv")
    
    return X_clean, y_clean, outlier_indices


if __name__ == "__main__":
    main()
