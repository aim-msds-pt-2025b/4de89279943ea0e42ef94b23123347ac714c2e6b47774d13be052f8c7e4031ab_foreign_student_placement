import os
import joblib
import warnings
from pathlib import Path
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

# Suppress joblib CPU core detection warnings on Windows
warnings.filterwarnings(
    "ignore", message=".*Could not find the number of physical cores.*"
)
warnings.filterwarnings(
    "ignore", message=".*The system cannot find the file specified.*"
)

# Set joblib to use logical cores instead of trying to detect physical cores
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

# Optional MLflow import
try:
    from mlflow_config import MLflowTracker
except ImportError:
    MLflowTracker = None

# suppress convergence warnings on LogisticRegression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class CustomMLModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper for your trained model.
    Encapsulates preprocessing and prediction logic.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None  # scaler, encoder, etc.
        self.feature_names = None

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        self.model = joblib.load(context.artifacts["model"])
        # Load preprocessor if exists
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        # Load feature names
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        # Apply preprocessing if available
        if self.preprocessor:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = model_input.values

        # Make predictions
        predictions = self.model.predict(processed_input)
        return predictions


def train_base_models(X_train, y_train, models_dir: str = "models"):
    """Train base models with MLflow tracking."""
    # Set MLflow tracking URI to local directory for testing
    mlflow.set_tracking_uri("file:./mlruns")

    os.makedirs(models_dir, exist_ok=True)
    models = {
        "randomforest": RandomForestClassifier(random_state=42),
        "logisticregression": LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=42, n_jobs=-1
        ),
        "gradientboosting": GradientBoostingClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(),
    }
    saved = {}

    for name, clf in models.items():
        with mlflow.start_run(run_name=f"train_{name}"):
            # Log exactly 3 hyperparameters for classification
            if name == "randomforest":
                mlflow.log_param("n_estimators", clf.n_estimators)
                mlflow.log_param("max_depth", clf.max_depth)
                mlflow.log_param("random_state", clf.random_state)
            elif name == "logisticregression":
                mlflow.log_param("solver", clf.solver)
                mlflow.log_param("max_iter", clf.max_iter)
                mlflow.log_param("random_state", clf.random_state)
            elif name == "gradientboosting":
                mlflow.log_param("n_estimators", clf.n_estimators)
                mlflow.log_param("max_depth", clf.max_depth)
                mlflow.log_param("random_state", clf.random_state)
            elif name == "svm":
                mlflow.log_param("C", clf.C)
                mlflow.log_param("kernel", clf.kernel)
                mlflow.log_param("random_state", clf.random_state)
            elif name == "knn":
                mlflow.log_param("n_neighbors", clf.n_neighbors)
                mlflow.log_param("weights", clf.weights)
                mlflow.log_param("algorithm", clf.algorithm)

            # Train model
            clf.fit(X_train, y_train)

            # Save model locally
            path = os.path.join(models_dir, f"model_{name}.pkl")
            joblib.dump(clf, path)
            saved[name] = path

            # Log model using custom PyFunc wrapper
            artifacts = {
                "model": path,
            }

            # Save feature names for the custom model
            feature_names_path = os.path.join(models_dir, f"feature_names_{name}.txt")
            with open(feature_names_path, "w") as f:
                for feature in X_train.columns:
                    f.write(f"{feature}\n")
            artifacts["feature_names"] = feature_names_path

            # Log custom model to MLflow
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CustomMLModel(),
                artifacts=artifacts,
                pip_requirements=["scikit-learn", "pandas", "numpy", "joblib"],
            )

            # Save artifacts to mlflow/artifacts/
            mlflow_artifacts_dir = "mlflow/artifacts"
            os.makedirs(mlflow_artifacts_dir, exist_ok=True)
            mlflow_model_path = os.path.join(mlflow_artifacts_dir, f"model_{name}.pkl")
            joblib.dump(clf, mlflow_model_path)

    return saved
    return saved


def tune_models(X_train, y_train, X_test=None, y_test=None, track_mlflow: bool = True):
    """Tune hyperparameters and optionally track with MLflow."""
    param_grids = {
        "randomforest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "gradientboosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
        "logisticregression": {
            # Use lbfgs with L2 only to avoid incompatible/slow combos and warnings
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [2000],
        },
        "knn": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
    }
    base = {
        "randomforest": RandomForestClassifier(random_state=42),
        "gradientboosting": GradientBoostingClassifier(random_state=42),
        # n_jobs is ignored by lbfgs but kept harmlessly; set solver via grid
        "logisticregression": LogisticRegression(random_state=42, n_jobs=-1),
        "knn": KNeighborsClassifier(),
    }
    # Lighter CV to speed up runs
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best = {}

    # Initialize MLflow tracker if requested
    mlflow_tracker = MLflowTracker() if (track_mlflow and MLflowTracker) else None

    for name, model in base.items():
        rs = RandomizedSearchCV(
            model,
            param_distributions=param_grids[name],
            n_iter=6,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            error_score="raise",
        )
        rs.fit(X_train, y_train)
        best[name] = rs.best_estimator_

        # Track with MLflow if enabled and test data provided
        if (
            track_mlflow
            and mlflow_tracker
            and X_test is not None
            and y_test is not None
        ):
            try:
                run_id = mlflow_tracker.log_model_run(
                    model_name=name,
                    model=rs.best_estimator_,
                    X_test=X_test,
                    y_test=y_test,
                    hyperparams=rs.best_params_,
                    additional_metrics={
                        "cv_score": rs.best_score_,
                        "cv_std": rs.cv_results_["std_test_score"][rs.best_index_],
                    },
                )
                print(f"MLflow run logged for {name}: {run_id}")
            except Exception as e:
                print(f"Warning: Could not log {name} to MLflow: {e}")

    return best


def build_ensemble(best_estimators, X_train, y_train, models_dir: str = "models"):
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


# --- Helpers used by run_pipeline_new.py ---
def tune_logistic_regression(X, y):
    param_grid = {
        "C": [0.01, 0.1, 1, 3, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "class_weight": [None],
    }
    lr = LogisticRegression(max_iter=1000)
    gs = GridSearchCV(
        lr,
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_score_


def train_dummy_baseline(X, y):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X, y)
    return dummy


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
