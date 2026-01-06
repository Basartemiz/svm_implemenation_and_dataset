# Fruit and Vegetable Classification with SVM

This repository contains implementations of various machine learning classifiers for fruit and vegetable image classification, with a focus on Support Vector Machine (SVM) implementation from scratch.

## Dataset

- **Classes**: 5 classes (apple, banana, carrot, orange, tomato)
- **Total Images**: ~670 images
- **Image Size**: 64x64 RGB
- **Features**: Flattened pixel values (12,288 dimensions before PCA)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:

- scikit-learn
- scikit-optimize (for Bayesian hyperparameter search)
- numpy
- matplotlib
- pandas
- quadprog (for SVM optimization)
- hdbscan (for clustering-based outlier detection)

## Project Structure

```
.
├── classifiers/              # Classifiers without PCA
├── classifiers_pca/         # Classifiers with PCA (main implementations)
│   ├── svm_classifier.py           # SVM from scratch with analysis
│   ├── svm_outlier_detection.py    # SVM-based outlier detection (Task 1.4)
│   ├── logistic_regression.py      # Linear logistic regression
│   ├── logistic_regression_nonLinear.py  # Non-linear logistic regression
│   ├── k_nearest_neighbors.py      # k-NN classifier
│   ├── naive_bayes.py              # Naive Bayes classifier
│   ├── random_forest.py            # Random Forest classifier
│   ├── pca.py                      # PCA exploration (Task 2.1)
│   └── clustering.py                # Clustering analysis (Task 2.2, 2.3)
├── svm/                     # Custom SVM implementation from scratch
│   └── svm.py
├── data/                    # Data loading and preprocessing
│   ├── preprocess.py        # Data loading functions
│   └── detect_outliers.py   # Clustering-based outlier detection
└── results/                 # Output directory for results and plots
```

## How to Reproduce Results

### Task 1: Classification

#### 1.1 All Classifiers with Training Times

Run each classifier to get training times and hyperparameters:

```bash
# Logistic Regression (Linear)
python classifiers_pca/logistic_regression.py

# Logistic Regression (Non-linear)
python classifiers_pca/logistic_regression_nonLinear.py

# SVM (from scratch)
python classifiers_pca/svm_classifier.py

# k-Nearest Neighbors
python classifiers_pca/k_nearest_neighbors.py

# Naive Bayes
python classifiers_pca/naive_bayes.py

# Random Forest
python classifiers_pca/random_forest.py
```

**Note**: All classifiers use:

- `test_size=0.2` for consistent comparison
- `StandardScaler` for feature normalization
- PCA with 95% variance retention (except k-NN and Naive Bayes which use original features)
- Bayesian hyperparameter search (BayesSearchCV) instead of grid search

#### 1.2 SVM Analysis

The SVM classifier (`svm_classifier.py`) includes:

- Combined support vector visualization (all classes in one plot)
- Farthest point identification and visualization
- Support vector pairwise distance analysis
- Comparison of closest SV pairs with confusion matrix

```bash
python classifiers_pca/svm_classifier.py
```

#### 1.3 Performance Comparison

After running all classifiers, compare their results:

- Training times are printed to console
- Hyperparameters are printed to console
- Results are saved to CSV files in `results/` directory

#### 1.4 SVM-based Outlier Detection

```bash
python classifiers_pca/svm_outlier_detection.py
```

This implements outlier detection using SVM slack variables:

- Trains soft-margin SVM
- Computes slack variables: ξᵢ = max(0, 1 - yᵢ(w·xᵢ + b))
- Identifies outliers as points with slack > threshold (95th percentile)
- Generates visualizations of detected outliers

**Note**: A demo video (5-10 minutes) is required for Task 1.4(b). Record a video showing:

- Brief introduction with framework illustration
- Evaluation protocol explanation
- Live demo of detected outliers

### Task 2: Unsupervised Learning

#### 2.1 PCA Exploration

```bash
python classifiers_pca/pca.py
```

This generates:

- Cumulative explained variance plot
- Reconstruction error vs. number of components plot
- Analysis of components needed for 80%, 90%, and 95% variance retention

**Important**: After PCA exploration, re-run all classifiers from Task 1.1 with PCA-reduced features to compare performance.

#### 2.2 Clustering Evaluation

```bash
python classifiers_pca/clustering.py
```

This performs:

- KMeans clustering after PCA reduction
- Evaluation using Silhouette Score (internal metric)
- Evaluation using Adjusted Rand Index (external metric)
- Visualization of clusters

#### 2.3 Outlier Detection Comparison

The clustering script also compares:

- Clustering-based outlier detection (HDBSCAN + KMeans)
- SVM-based outlier detection (slack variables)

Comparison results are saved to `results/outlier_detection_comparison.csv`

## Hyperparameter Search

All classifiers use **Bayesian Optimization** (BayesSearchCV from scikit-optimize) instead of grid search for more efficient hyperparameter tuning.

### Hyperparameter Ranges

- **SVM**:

  - C: [1e-4, 1e4] (log-uniform)
  - Kernel: linear, rbf, poly
  - gamma (rbf/poly): [1e-4, 1e4] (log-uniform)
  - degree (poly): [2, 5]
  - r (poly): [0.1, 10] (log-uniform)

- **Logistic Regression**:

  - C: [1e-4, 1e4] (log-uniform)
  - solver: liblinear, lbfgs

- **k-NN**:

  - n_neighbors: [1, 30]
  - weights: uniform, distance
  - metric: euclidean, manhattan

- **Naive Bayes**:

  - var_smoothing: [1e-9, 1e0] (log-uniform)

- **Random Forest**:
  - n_estimators: [50, 200]
  - max_depth: None, 10, 20, 30
  - min_samples_split: [2, 10]
  - min_samples_leaf: [1, 4]

## Output Files

All results are saved to the `results/` directory:

- `*_results.csv`: Hyperparameter search results for each classifier
- `svm_classifier_results_pca.csv`: SVM results
- `pca_exploration.png`: PCA variance and reconstruction error plots
- `svm_outlier_detection.png`: SVM outlier detection visualization
- `outlier_detection_comparison.csv`: Comparison of outlier detection methods
- `pca_exploration_results.csv`: PCA analysis summary

## Custom SVM Implementation

The custom SVM implementation (`svm/svm.py`) includes:

- Binary SVM with linear, RBF, and polynomial kernels
- Multi-class SVM using one-vs-rest approach
- Support vector extraction
- Farthest point identification
- Compatible with scikit-learn's Pipeline and BayesSearchCV

## Notes

1. **Consistency**: All classifiers use `test_size=0.2` and `StandardScaler` for fair comparison
2. **PCA**: Most classifiers use PCA with 95% variance retention (k-NN and Naive Bayes use original features)
3. **Outlier Detection**:
   - Task 1.4 requires SVM-based outlier detection (slack variables)
   - Task 2.3 compares clustering-based (HDBSCAN/KMeans) with SVM-based methods
4. **Training Times**: All classifiers print training time to console
5. **Hyperparameters**: All hyperparameters are determined using Bayesian search with 5-fold cross-validation

## Troubleshooting

- If you encounter import errors, ensure the project root is in your Python path
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Results directory is created automatically if it doesn't exist
- For large datasets, Bayesian search may take time; adjust `n_iter` parameter if needed
- **Suppressing warnings**: To suppress pandas bottleneck warnings, run scripts with: `python -W ignore classifiers_pca/script_name.py`
  - Alternatively, upgrade bottleneck: `pip install --upgrade bottleneck`

## Contact

For questions or issues, please refer to the assignment documentation or contact the course instructor.
