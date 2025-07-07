import os
import joblib
import warnings
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# suppress convergence warnings on LogisticRegression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def train_base_models(X_train, y_train, models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    models = {
        "randomforest": RandomForestClassifier(random_state=42),
        "logisticregression": LogisticRegression(
            solver="saga", max_iter=5000, random_state=42, n_jobs=-1
        ),
        "gradientboosting": GradientBoostingClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(),
    }
    saved = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        path = os.path.join(models_dir, f"model_{name}.pkl")
        joblib.dump(clf, path)
        saved[name] = path
    return saved


def tune_models(X_train, y_train):
    param_grids = {
        "randomforest": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
        "gradientboosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "logisticregression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": [None, "l2", "l1"],
            "solver": ["saga"],
            "max_iter": [2000],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
    }
    base = {
        "randomforest": RandomForestClassifier(random_state=42),
        "gradientboosting": GradientBoostingClassifier(random_state=42),
        "logisticregression": LogisticRegression(random_state=42, n_jobs=-1),
        "knn": KNeighborsClassifier(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best = {}
    for name, model in base.items():
        rs = RandomizedSearchCV(
            model,
            param_distributions=param_grids[name],
            n_iter=10,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            error_score="raise",
        )
        rs.fit(X_train, y_train)
        best[name] = rs.best_estimator_
    return best


def build_ensemble(best_estimators, X_train, y_train, models_dir="models"):
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in best_estimators.items()],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "model_ensemble.pkl")
    joblib.dump(ensemble, path)
    return ensemble
