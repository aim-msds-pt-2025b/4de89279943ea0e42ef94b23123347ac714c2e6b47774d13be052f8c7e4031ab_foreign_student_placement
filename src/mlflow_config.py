"""MLflow configuration and utilities."""

import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow import MlflowClient
from typing import Dict, Optional
import inspect
import sys
import pandas as pd
import warnings
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Suppress MLflow deprecation warnings
warnings.filterwarnings("ignore", message=".*artifact_path.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# Also suppress MLflow logging warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)


class MLflowTracker:
    """MLflow experiment tracking utilities."""

    def __init__(
        self,
        experiment_name: str = "student_placement_prediction",
        tracking_uri: str = "mlruns",
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (local by default)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = None
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            print(f"Warning: Could not create experiment: {e}")

        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()

    def log_model_run(
        self,
        model_name: str,
        model,
        X_test,
        y_test,
        hyperparams: Optional[Dict] = None,
        additional_metrics: Optional[Dict] = None,
        artifacts_dir: str = "models",
    ) -> str:
        """Log a model run to MLflow.

        Args:
            model_name: Name of the model
            model: Trained model object
            X_test: Test features
            y_test: Test labels
            hyperparams: Model hyperparameters
            additional_metrics: Additional metrics to log
            artifacts_dir: Directory to save model artifacts

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            # Log parameters
            if hyperparams:
                mlflow.log_params(hyperparams)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            # Calculate and log metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }

            if y_pred_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

            if additional_metrics:
                metrics.update(additional_metrics)

            mlflow.log_metrics(metrics)

            # Create input example for model signature (sample of test data)
            # Handle both pandas DataFrame and numpy array inputs
            if hasattr(X_test, "iloc"):
                # DataFrame case
                input_example = X_test.iloc[:5]
            elif hasattr(X_test, "shape"):
                # Numpy array case
                input_example = X_test[:5]
            else:
                # Fallback
                input_example = None

            # Prepare stable environment spec to avoid env inference warnings
            try:
                import sklearn as _sk
                import pandas as _pd
                import numpy as _np
                import cloudpickle as _cp
                import mlflow as _mf

                pip_requirements = [
                    f"scikit-learn=={_sk.__version__}",
                    f"pandas=={_pd.__version__}",
                    f"numpy=={_np.__version__}",
                    f"cloudpickle=={_cp.__version__}",
                    f"mlflow=={_mf.__version__}",
                ]
            except Exception:
                # Fall back to unpinned minimal set
                pip_requirements = [
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "cloudpickle",
                    "mlflow",
                ]

            # Build a minimal conda env to avoid MLflow trying to resolve pip version
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            conda_env = {
                "name": "mlflow-env",
                "channels": ["conda-forge"],
                "dependencies": [
                    f"python={py_ver}.*",
                    "pip",
                    {"pip": pip_requirements},
                ],
            }

            # Prepare kwargs with feature-detection for MLflow version compatibility
            log_model_sig = inspect.signature(mlflow.sklearn.log_model)
            kwargs: Dict[str, object] = {"sk_model": model}
            if "name" in log_model_sig.parameters:
                kwargs["name"] = "model"
            else:
                # Backward-compatible fallback; may emit deprecation warning on newer MLflow
                kwargs["artifact_path"] = "model"
            if (
                input_example is not None
                and "input_example" in log_model_sig.parameters
            ):
                kwargs["input_example"] = input_example
            # Provide exactly one env spec: prefer pip_requirements if supported, else conda_env
            if "pip_requirements" in log_model_sig.parameters:
                kwargs["pip_requirements"] = pip_requirements
            elif "conda_env" in log_model_sig.parameters:
                kwargs["conda_env"] = conda_env
            if "env_manager" in log_model_sig.parameters:
                # Prefer virtualenv to avoid conda on some setups
                kwargs["env_manager"] = "virtualenv"

            # Log model (robust to different MLflow versions)
            model_info = mlflow.sklearn.log_model(**kwargs)

            # Register the model separately (works across versions)
            mlflow.register_model(
                model_uri=model_info.model_uri, name=f"student_placement_{model_name}"
            )

            # Log model info
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("model_name", model_name)

            return run.info.run_id

    def register_best_model(self, model_name: str, run_id: str, stage: str = "Staging"):
        """Register the best model to Model Registry.

        Args:
            model_name: Name of the registered model
            run_id: MLflow run ID
            stage: Model stage (Staging, Production, etc.)
        """
        model_uri = f"runs:/{run_id}/model"
        registered_model_name = f"student_placement_{model_name}"

        # Register model
        model_version = mlflow.register_model(model_uri, registered_model_name)

        # Transition to stage
        self.client.transition_model_version_stage(
            name=registered_model_name, version=model_version.version, stage=stage
        )

        return model_version

    def get_model_by_stage(self, model_name: str, stage: str = "Production"):
        """Load model from registry by stage.

        Args:
            model_name: Name of the registered model
            stage: Model stage

        Returns:
            Loaded model
        """
        registered_model_name = f"student_placement_{model_name}"
        model_uri = f"models:/{registered_model_name}/{stage}"
        return mlflow.sklearn.load_model(model_uri)

    def compare_runs(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """Compare all runs in the experiment.

        Args:
            experiment_id: MLflow experiment ID

        Returns:
            DataFrame with run comparison
        """
        if experiment_id is None:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id

        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        return runs[
            [
                "run_id",
                "status",
                "start_time",
                "metrics.accuracy",
                "metrics.precision",
                "metrics.recall",
                "metrics.f1_score",
                "metrics.roc_auc",
                "params.model_name",
            ]
        ].sort_values("metrics.accuracy", ascending=False)
