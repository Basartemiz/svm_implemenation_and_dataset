try:
    from data.preprocess import load_data
    from data.detect_outliers import detect_outliers_isolation_forest
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

    print(f"Original data shape: {X.shape}")

    # split data into train and test sets
    from sklearn.model_selection import train_test_split

    X,y=detect_outliers_isolation_forest(X, y) # detect and remove outliers

    print(f"Data shape after outlier removal: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    #use a pipeline to standardize the data then apply logistic regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("log_reg", LogisticRegression(max_iter=1000)),
        ]
    )

    parameters={
        'log_reg__C':np.logspace(-4,4,10),
        'log_reg__solver': ['liblinear', 'lbfgs']
    }

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_parameters={'log_reg__C': np.float64(0.000774263682681127), 'log_reg__solver': 'lbfgs'}
    pipeline.set_params(**best_model_parameters)
    best_model=pipeline
    from sklearn.model_selection import GridSearchCV
    if(best_model is None):
        clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, scoring='accuracy')
        clf.fit(X_train, y_train)
    else:
        import time
        start_time = time.time()
        best_model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time for best model: {training_time:.2f} seconds")
        print(f"Accuracy Score : {accuracy_score(y_test, best_model.predict(X_test))*100:.2f}%")
        #get train the best model on non outlier data and measure time
        import time
        X,y=load_data(data_path, image_size=(64, 64))
        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
            )
        start_time = time.time()
        best_model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time for best model: {training_time:.2f} seconds")
        print(f"Accuracy Score : {accuracy_score(y_test, best_model.predict(X_test))*100:.2f}%")
        return 
        
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

    #get train the best model on non outlier data and measure time
    import time
    X,y=load_data(data_path, image_size=(64, 64))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    start_time = time.time()
    best_model = clf.best_estimator_
    best_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time for best model: {training_time:.2f} seconds")
    print(f"Accuracy Score : {accuracy_score(y_test, best_model.predict(X_test))*100:.2f}%")


    #inspect results of all models
    results = clf.cv_results_
    results_sorted = sorted(
        zip(results["mean_test_score"], results["params"]), reverse=True
    )
    print("\nAll model results (sorted by accuracy):")
    for mean_score, params in results_sorted:
        print(f"Accuracy: {mean_score:.4f} | Parameters: {params}")
    
    #save results to a csv file
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/logistic_regression_results.csv", index=False)


if __name__ == "__main__":
    main()
