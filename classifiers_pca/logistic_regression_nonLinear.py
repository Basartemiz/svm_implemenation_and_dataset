import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from data.preprocess import load_data
except ImportError:  
    import os
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from data.preprocess import load_data


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # get data (X,y)
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "fruit_images")
    X, y = load_data(data_path, image_size=(64, 64))

    # split data into train and test sets
    from sklearn.model_selection import train_test_split


    #use pca to reduce dimensionality
    X=X.reshape(X.shape[0], -1)  # Flatten images for PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)  # retain 95% of variance (consistent with other classifiers)
    X = pca.fit_transform(X)
    print(f"PCA reduced shape: {X.shape}")  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #use a pipeline to standardize the data then apply logistic regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import PolynomialFeatures
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures()),
            ("log_reg", LogisticRegression(max_iter=1000)),
        ]
    )

    parameters={
        'poly__degree':[2,3], #try degrees 2 and 3
        'log_reg__C':np.logspace(-4,4,10),
        'log_reg__solver': ['liblinear', 'lbfgs']
    }

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    
    param_grid = {
        'poly__degree': Integer(2, 3),
        'log_reg__C': Real(1e-4, 1e4, prior='log-uniform'),
        'log_reg__solver': Categorical(['liblinear', 'lbfgs'])
    }

    clf = BayesSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', n_iter=30, random_state=42)
    clf.fit(X_train, y_train)

    #see results

    print(f"\nBest parameters found: {clf.best_params_}")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy * 100:.2f}%")


    #get training time using best estimator
    import time
    start_time = time.time()
    best_model = clf.best_estimator_
    best_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time for best model: {training_time:.2f} seconds")


    #inspect results of all models
    results = clf.cv_results_
    results_sorted = sorted(
        zip(results["mean_test_score"], results["params"]), reverse=True,
        key=lambda x: x[0]
    )
    print("\nTop 10 model results (sorted by accuracy):")
    for idx, (mean_score, params) in enumerate(results_sorted[:10], 1):
        print(f"{idx:2d}. Accuracy: {mean_score:.4f} | degree={params['poly__degree']}, C={params['log_reg__C']:.4f}, solver={params['log_reg__solver']}")
    

if __name__ == "__main__":
    main()
