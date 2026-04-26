import numpy as np
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

SEED = 42
N_PCA = 30  # covers >99% of variance on this dataset


def _make_pipe(clf, use_pca=True, n_pca=N_PCA):
    """Helper to avoid repeating the scaler+PCA boilerplate."""
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_pca, random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def build_baseline_pipeline(n_components=30):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=SEED)),
        ("svm", SVC(kernel="rbf", random_state=SEED)),
    ])


def build_candidate_models():
    """All the classifiers we want to compare."""
    models = {
        "Gaussian Naive Bayes": _make_pipe(GaussianNB()),
        "QDA": _make_pipe(QuadraticDiscriminantAnalysis(), n_pca=20),
        "LDA (Fisher)": _make_pipe(LinearDiscriminantAnalysis(), use_pca=False),
        "Logistic Regression": _make_pipe(
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED)
        ),
        "K-Nearest Neighbours": _make_pipe(KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF kernel)": _make_pipe(
            SVC(kernel="rbf", probability=True, random_state=SEED)
        ),
        "Random Forest": _make_pipe(
            RandomForestClassifier(n_estimators=300, random_state=SEED)
        ),
        "Gradient Boosting": _make_pipe(
            GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.1,
                max_depth=3, random_state=SEED,
            )
        ),
        "MLP Neural Network": _make_pipe(
            MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", solver="adam",
                alpha=1e-4, learning_rate="adaptive",
                max_iter=500, early_stopping=True,
                random_state=SEED,
            ),
            use_pca=False,  # MLP can handle the full feature space
        ),
    }
    return models


def tune_svm(X_train, y_train, cv=5):
    """Grid search over PCA components, C, and gamma for the SVM pipeline."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=SEED)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=SEED)),
    ])
    param_grid = {
        "pca__n_components": [10, 20, 30, 40],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.001],
    }
    gs = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="accuracy",
        n_jobs=-1, verbose=1, return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs


def tune_mlp(X_train, y_train, cv=5, n_iter=20):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            solver="adam", activation="relu",
            early_stopping=True, max_iter=500,
            random_state=SEED)),
    ])
    param_dist = {
        "mlp__hidden_layer_sizes": [
            (128,), (256,), (128, 64), (256, 128),
            (256, 128, 64), (512, 256, 128),
        ],
        "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
    }
    rs = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=n_iter, cv=cv, scoring="accuracy",
        n_jobs=-1, verbose=1, random_state=SEED,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)
    return rs


def build_ensemble(svm_best=None, mlp_best=None):
    """
    Soft-voting ensemble. If tuned models are provided, use those;
    otherwise fall back to default configurations.
    """
    estimators = [
        ("lr", _make_pipe(
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED)
        )),
        ("rf", _make_pipe(
            RandomForestClassifier(n_estimators=300, random_state=SEED)
        )),
        ("gb", _make_pipe(
            GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED)
        )),
    ]

    if svm_best is not None:
        estimators.append(("svm", svm_best))
    else:
        estimators.append(("svm", _make_pipe(
            SVC(kernel="rbf", probability=True, random_state=SEED)
        )))

    if mlp_best is not None:
        estimators.append(("mlp", mlp_best))
    else:
        estimators.append(("mlp", _make_pipe(
            MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", solver="adam",
                max_iter=500, early_stopping=True,
                random_state=SEED,
            ),
            use_pca=False,
        )))

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
