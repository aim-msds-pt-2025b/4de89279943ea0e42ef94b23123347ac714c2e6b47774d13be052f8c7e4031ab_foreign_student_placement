"""HW3: Five-task ML pipeline with drift-driven branching."""

from datetime import datetime
import os
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# Ensure MLflow URI is available in tasks
os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

DATA_PATH = "/opt/airflow/data/global_student_migration.csv"
PROC_DIR = "/opt/airflow/data/processed"
REPORTS_DIR = "/opt/airflow/reports"
MODELS_DIR = "/opt/airflow/models"

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_base_models
from evaluation import evaluate_models
from drift_detection import DriftDetector


def preprocess_task(**ctx):
    os.makedirs(PROC_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
    X_train.to_pickle(f"{PROC_DIR}/X_train.pkl")
    X_test.to_pickle(f"{PROC_DIR}/X_test.pkl")
    y_train.to_pickle(f"{PROC_DIR}/y_train.pkl")
    y_test.to_pickle(f"{PROC_DIR}/y_test.pkl")


def feature_engineering_task(**ctx):
    X_train = pd.read_pickle(f"{PROC_DIR}/X_train.pkl")
    X_test = pd.read_pickle(f"{PROC_DIR}/X_test.pkl")
    Xtr_fe, Xte_fe = engineer_features(X_train, X_test)
    Xtr_fe.to_pickle(f"{PROC_DIR}/X_train_fe.pkl")
    Xte_fe.to_pickle(f"{PROC_DIR}/X_test_fe.pkl")


def train_task(**ctx):
    os.makedirs(MODELS_DIR, exist_ok=True)
    Xtr = pd.read_pickle(f"{PROC_DIR}/X_train_fe.pkl")
    ytr = pd.read_pickle(f"{PROC_DIR}/y_train.pkl")
    train_base_models(Xtr, ytr, models_dir=MODELS_DIR)


def evaluate_task(**ctx):
    Xte = pd.read_pickle(f"{PROC_DIR}/X_test_fe.pkl")
    yte = pd.read_pickle(f"{PROC_DIR}/y_test.pkl")
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="most_frequent").fit(Xte, yte)
    df = evaluate_models({"dummy": dummy}, Xte, yte)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(f"{REPORTS_DIR}/eval_metrics.csv", index=True)


def drift_detection_task(**ctx):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    raw["placement_status"] = raw["placement_status"].map({"Placed": 1, "Not Placed": 0})
    det = DriftDetector(
        reference_data=raw,
        target_column="placement_status",
        numerical_features=["gpa_or_score", "test_score", "year_of_enrollment", "graduation_year"],
        categorical_features=["origin_country", "destination_country", "field_of_study"],
    )
    # Use test split as a simple current snapshot
    cur = pd.read_pickle(f"{PROC_DIR}/X_test.pkl").copy()
    cur["placement_status"] = pd.read_pickle(f"{PROC_DIR}/y_test.pkl")
    res = det.detect_data_drift(cur)
    det.save_drift_report(res, output_path=f"{REPORTS_DIR}/drift_report.json")
    # Decide drift
    drifted = bool(res.get("dataset_drift_detected", False) or (res.get("drift_share", 0) > 0.3))
    return {"drifted": drifted}


def decide_branch(**ctx):
    info = ctx["ti"].xcom_pull(task_ids="drift_detection")
    return "retrain_model" if info and info.get("drifted") else "pipeline_complete"


with DAG(
    dag_id="hw3_branching_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["hw3", "mlflow", "drift"],
) as dag:
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_task)
    fe = PythonOperator(task_id="feature_engineering", python_callable=feature_engineering_task)
    train = PythonOperator(task_id="train_model", python_callable=train_task)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_task)
    drift = PythonOperator(task_id="drift_detection", python_callable=drift_detection_task)
    branch = BranchPythonOperator(task_id="decide_retrain", python_callable=decide_branch)
    retrain = PythonOperator(task_id="retrain_model", python_callable=train_task)
    done = EmptyOperator(task_id="pipeline_complete")

    preprocess >> fe >> train >> evaluate >> drift >> branch
    branch >> retrain >> done
    branch >> done
