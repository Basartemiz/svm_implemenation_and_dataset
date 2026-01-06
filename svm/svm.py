#this code is for svm model implementation from scratch using numpy and quadprog for optimization
import numpy as np
import quadprog


class SVM:
    def __init__(self,X,Y, C=1.0, kernel='linear', degree=3, gamma=0.1, r=1):
        self.C = C
        self.w = None
        self.b = None
        self.X = X
        self.Y = Y
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.alphas = None
        self.full_alphas = None
        self._poly_norm_train = None

        self.indices = None #indices of farthest points
        self.kernel = kernel

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have compatible shapes")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must be finite (no NaN/inf)")
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be a positive finite number")

        n_samples, _ = X.shape
        self.X = X
        self.Y = y

        # build the P matrix for quadratic programming
        if self.kernel == 'linear':
            K = self.linear_kernel(X, y)
        elif self.kernel == 'poly':
            K = self.poly_kernel(X, y, degree=self.degree, gamma=self.gamma, r=self.r)
        elif self.kernel == 'rbf':
            K = self.rbf_kernel(X, y, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        if not np.isfinite(K).all():
            raise ValueError("Kernel matrix contains NaN/inf")

        # Polynomial kernels can become extremely ill-conditioned on high-dimensional inputs.
        # Normalizing by the diagonal preserves PSD while improving numerical stability.
        if self.kernel == "poly":
            diag = np.diag(K)
            diag = np.clip(diag, 1e-12, None)
            denom = np.sqrt(diag)
            self._poly_norm_train = denom
            K = K / (denom[:, None] * denom[None, :])
            K = (K + K.T) / 2.0

        P = np.outer(y, y) * K
        P = (P + P.T) / 2.0
        # quadprog minimizes: 1/2 x^T G x - a^T x
        # SVM dual (soft margin) minimizes: 1/2 a^T P a - 1^T a
        a = np.ones(n_samples, dtype=np.float64)

        # Numerical stability: rescale objective (does not change the minimizer)
        P_scale = float(np.max(np.abs(P))) if P.size else 1.0
        if not np.isfinite(P_scale) or P_scale <= 0:
            P_scale = 1.0
        G = P / P_scale
        a = a / P_scale

        # constraints
        I = np.eye(n_samples)
        Cmat = np.column_stack([y, I, -I])  # shape (N, 1+N+N)
        Cmat = np.asfortranarray(Cmat, dtype=np.float64)
        bvec = np.hstack([0.0, np.zeros(n_samples), -self.C * np.ones(n_samples, dtype=np.float64)])
        bvec = np.asarray(bvec, dtype=np.float64)

        # quadprog requires G (P) to be positive definite; add diagonal jitter if needed.
        jitter = 1e-12
        last_err = None
        alphas = None
        for _ in range(12):
            try:
                alphas = quadprog.solve_qp(G + jitter * np.eye(n_samples), a, Cmat, bvec, 1)[0]
                break
            except ValueError as e:
                last_err = e
                msg = str(e).lower()
                # Both errors can appear when the kernel matrix is extremely ill-conditioned.
                if "positive definite" in msg or "constraints are inconsistent" in msg:
                    jitter *= 10.0
                    continue
                raise
        if alphas is None:
            # Keep BayesSearchCV/GridSearchCV running: return a degenerate (all-zero) solution.
            alphas = np.zeros(n_samples, dtype=np.float64)

        full_alphas = alphas.copy()  # get the full alphas for later use
        self.full_alphas = full_alphas

        # support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv_X = X[sv]  # get support vectors
        self.sv_y = y[sv]  # get support vector labels

        # handle degenerate solutions (e.g., very small C) without crashing GridSearchCV
        if self.alphas.size == 0:
            self.w = None
            self.b = 0.0
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == -1)[0]
            if pos_indices.size == 0 or neg_indices.size == 0:
                self.indices = None
            else:
                self.indices = (int(pos_indices[0]), int(neg_indices[0]))
            return self

        # compute the decision boundary (only meaningful for linear kernel)
        if self.kernel == "linear":
            self.w = np.dot((self.alphas * self.sv_y), self.sv_X)

        # compute the bias term
        self.b = 0
        for n in range(len(self.alphas)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[n], sv])
        self.b /= len(self.alphas)

        # now get the farthest points
        distances = y * (np.dot(K, full_alphas * y) + self.b)
        # get indices of two farthest points of each class
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == -1)[0]
        if pos_indices.size == 0 or neg_indices.size == 0:
            self.indices = None
        else:
            farthest_pos = pos_indices[np.nanargmax(distances[pos_indices])]
            farthest_neg = neg_indices[np.nanargmax(distances[neg_indices])]
            self.indices = (int(farthest_pos), int(farthest_neg))
        return self


    #gets the kernel gram matrix
    def linear_kernel(self, X, y):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            K = X @ X.T
        return K

    def poly_kernel(self, X, y, degree=1, gamma=1.0, r=1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            K = (gamma * (X @ X.T) + r) ** degree
        return K


    def rbf_kernel(self, X, y,gamma):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            X_norm = np.sum(X**2, axis=1)
            sq_dists = X_norm[:, None] + X_norm[None, :] - 2 * (X @ X.T)
            return np.exp(-gamma * sq_dists)

       
    
    def predict(self, X_test):
        if self.full_alphas is None:
            raise ValueError("Model is not fitted yet")
        X_test = np.asarray(X_test, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.kernel == 'linear':
                K = self.X @ X_test.T
            elif self.kernel == 'poly':
                K = (self.gamma * (self.X @ X_test.T) + self.r) ** self.degree
                train_denom = self._poly_norm_train
                if train_denom is None:
                    diag_train = (self.gamma * np.sum(self.X**2, axis=1) + self.r) ** self.degree
                    train_denom = np.sqrt(np.clip(diag_train, 1e-12, None))
                diag_test = (self.gamma * np.sum(X_test**2, axis=1) + self.r) ** self.degree
                test_denom = np.sqrt(np.clip(diag_test, 1e-12, None))
                K = K / (train_denom[:, None] * test_denom[None, :])
            elif self.kernel == 'rbf':
                K = (-2 * np.dot(self.X, X_test.T) + np.sum(self.X**2, axis=1)[:, np.newaxis] + np.sum(X_test**2, axis=1)[np.newaxis, :])
                K = np.exp(-self.gamma * K)
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")

        if not np.isfinite(K).all():
            raise ValueError("Kernel matrix contains NaN/inf")


        y_predict = np.dot((self.full_alphas *self.Y), K) + self.b
        return y_predict #return raw decision function values then compare to understand class labels   



class MultiClassSVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=0.1, r=1):
        self.C = C
        self.models = {}
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.farthest_indices = {}
        self.farthest_points = {}
        self.support_vectors = {}
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.classes = self.classes_
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            model = SVM(X, y_binary, C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, r=self.r)
            model.fit(X, y_binary)
            self.models[cls] = model
            if model.indices is not None:
                self.farthest_indices[cls] = model.indices[0] #store farthest positive point index for each class
                self.farthest_points[cls] = model.X[model.indices[0]]
            else:
                self.farthest_indices[cls] = None
                self.farthest_points[cls] = None

            # store only positive-class support vectors for per-class inspection/visualization
            pos_sv = model.sv_y == 1
            self.support_vectors[cls] = model.sv_X[pos_sv]
        return self
 

    #gets the predictions with top two classes
    def predict_(self, X_test):
        scores = np.zeros((X_test.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            model = self.models[cls]
            scores[:, idx] = model.predict(X_test)
        #get the top two classes for each sample
        top_two_classes = np.argsort(-scores, axis=1)[:, :2]
        predictions = []
        for i in range(X_test.shape[0]):
            #return two most probable classes
            predictions.append((self.classes_[top_two_classes[i,0]], self.classes_[top_two_classes[i,1]]))
        return predictions 
    
    #normal predict method returning only top class for compatibility with sklearn
    def predict(self, X_test):
        scores = np.zeros((X_test.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            model = self.models[cls]
            scores[:, idx] = model.predict(X_test)
        #get the top class for each sample
        top_classes = np.argmax(scores, axis=1)
        predictions = self.classes_[top_classes]
        return predictions
    #get_params and set_params methods for compatibility with sklearn's GridSearchCV
    def get_params(self, deep=True):
        return {"C": self.C, "kernel": self.kernel, "degree": self.degree, "gamma": self.gamma, "r": self.r}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    



#--- IGNORE ---
if __name__ == "__main__":
    # Example usage
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # For simplicity, use only two classes
    X = X[y != 2]
    y = y[y != 2]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = np.where(y_train == 1, 1, -1)
    y_test  = np.where(y_test  == 1, 1, -1)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    svm = SVM(X_train, y_train, C=1.0, kernel='linear')
    svm.fit(X_train, y_train)

    # Predict
    predictions = svm.predict(X_test)
    predicted_classes = np.sign(predictions)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")


    #now test multiclass svm
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target 
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    # Train MultiClass SVM
    mc_svm = MultiClassSVM(C=1.0, kernel='linear')
    mc_svm.fit(X_train, y_train)
    # Predict
    mc_predictions = mc_svm.predict(X_test)
    # Convert predictions to single class by taking the first predicted class
    mc_predicted_classes = np.array([pred[0] for pred in mc_predictions])
    # Calculate accuracy
    mc_accuracy = np.mean(mc_predicted_classes == y_test)
    print(f"MultiClass SVM Accuracy: {mc_accuracy * 100:.2f}%")
#--- IGNORE ---
