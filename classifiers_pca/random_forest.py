try:
    from data.preprocess import load_data
    from data.detect_outliers import detect_outliers
except ImportError:  
    import os
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from data.preprocess import load_data


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



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
    pca = PCA(n_components=0.95)  # retain 95% of variance
    X = pca.fit_transform(X)
    print(f"PCA reduced shape: {X.shape}")

   

    X,y=detect_outliers(X, y) # detect and remove outliers

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier()),
        ]
    )

    parameters={
       'rf__n_estimators': [50, 100, 200],
       'rf__max_depth': [None, 10, 20, 30],
       'rf__min_samples_split': [2, 5, 10],
       'rf__min_samples_leaf': [1, 2, 4]
    }

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, scoring='accuracy')
    clf.fit(X_train, y_train)

    #see results

    print("Best parameters found: ", clf.best_params_)
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
    zip(results["mean_test_score"], results["params"]),
    key=lambda x: x[0],  
    reverse=True
)
    print("\nAll model results (sorted by accuracy):")
    for mean_score, params in results_sorted:
        print(f"Accuracy: {mean_score:.4f} | Parameters: {params}")
    
    

    #save results to a csv file
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "random_forest_results_pca.csv"), index=False)

    #viualize pca
  

    
if __name__ == "__main__":
    main()
