import numpy as np
from typing import Dict, Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

def build_baseline_pipeline(n_components: int = 30) -> Pipeline:    # PCA + SVM baseline
    # k(x, x') = exp(−γ ‖x − x'‖²)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=42)),
        ("svm",    SVC(kernel="rbf", random_state=42)),
    ])

def build_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:    # Classification algorithms — Bishop (2006)
    models = {
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    GaussianNB()),
        ]),
        "QDA": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=20, random_state=random_state)),
            ("clf",    QuadraticDiscriminantAnalysis()),
        ]),
        "LDA (Fisher)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearDiscriminantAnalysis()),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    LogisticRegression(
                max_iter=5000, 
                solver="lbfgs", random_state=random_state)),
        ]),
        "K-Nearest Neighbours": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM (RBF kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    SVC(kernel="rbf", probability=True,
                          random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    RandomForestClassifier(
                n_estimators=300, random_state=random_state)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1,
                max_depth=5, random_state=random_state)),
        ]),
        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                random_state=random_state)),
        ]),
    }
    return models

def tune_svm(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> GridSearchCV:    # p(C_k | x) = p(x | C_k) p(C_k) / p(x)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(random_state=42)),
        ("svm",    SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    param_grid = {
        "pca__n_components": [10, 20, 30, 40],
        "svm__C":            [0.1, 1, 10, 100],
        "svm__gamma":        ["scale", "auto", 0.01, 0.001],
    }
    gs = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="accuracy",
        n_jobs=-1, verbose=1, return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs

def tune_mlp(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, n_iter: int = 20) -> RandomizedSearchCV:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            solver="adam", activation="relu",
            early_stopping=True, max_iter=500,
            random_state=42)),
    ])
    param_distributions = {
        "mlp__hidden_layer_sizes": [
            (128,), (256,), (128, 64), (256, 128),
            (256, 128, 64), (512, 256, 128),
        ],
        "mlp__alpha":         [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
    }
    rs = RandomizedSearchCV(
        pipe, param_distributions,
        n_iter=n_iter, cv=cv, scoring="accuracy",
        n_jobs=-1, verbose=1, random_state=42,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)
    return rs

def build_ensemble(svm_best: Optional[Pipeline] = None, 
                   mlp_best: Optional[Pipeline] = None, 
                   random_state: int = 42) -> VotingClassifier:
    # p(C_k | x) ≈ (1/M) Σ_m  p_m(C_k | x)
    estimators = [
        ("lr",  Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    LogisticRegression(max_iter=5000, solver="lbfgs", random_state=random_state)),
        ])),
        ("rf",  Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    RandomForestClassifier(n_estimators=300, random_state=random_state)),
        ])),
        ("gb",  Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    GradientBoostingClassifier(n_estimators=200, random_state=random_state)),
        ])),
    ]

    if svm_best is not None:
        estimators.append(("svm", svm_best))
    else:
        estimators.append(("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=random_state)),
            ("clf",    SVC(kernel="rbf", probability=True, random_state=random_state)),
        ])))

    if mlp_best is not None:
        estimators.append(("mlp", mlp_best))
    else:
        estimators.append(("mlp", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", solver="adam",
                max_iter=500, early_stopping=True,
                random_state=random_state)),
        ])))

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
