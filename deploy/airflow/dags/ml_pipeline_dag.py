# deploy/airflow/dags/ml_pipeline_dag.py

import os
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_base_models, tune_models, build_ensemble
from evaluation import evaluate_models, select_and_save_best
from visualization import (
    plot_target_distribution,
    plot_feature_correlations,
    plot_roc_curves,
    plot_confusion_matrix,
)

DATA_PATH   = "/opt/airflow/data/global_student_migration.csv"
PROC_DIR    = "/opt/airflow/data/processed"
MODELS_DIR  = "/opt/airflow/models"

def task_preprocess(**ctx):
    # 1) preprocess and split
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
    os.makedirs(PROC_DIR, exist_ok=True)
    # 2) write out as pickles
    X_train.to_pickle(f"{PROC_DIR}/X_train.pkl")
    X_test.to_pickle( f"{PROC_DIR}/X_test.pkl")
    y_train.to_pickle(f"{PROC_DIR}/y_train.pkl")
    y_test.to_pickle( f"{PROC_DIR}/y_test.pkl")
    # 3) push file‐paths only
    ti = ctx['ti']
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        ti.xcom_push(key=f"{name}_path", value=f"{PROC_DIR}/{name}.pkl")

def task_engineer(**ctx):
    ti = ctx['ti']
    X_train = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="X_train_path"))
    X_test  = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="X_test_path"))
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)
    # store engineered
    X_train_fe.to_pickle(f"{PROC_DIR}/X_train_fe.pkl")
    X_test_fe.to_pickle( f"{PROC_DIR}/X_test_fe.pkl")
    ti.xcom_push(key="X_train_fe_path", value=f"{PROC_DIR}/X_train_fe.pkl")
    ti.xcom_push(key="X_test_fe_path",  value=f"{PROC_DIR}/X_test_fe.pkl")

def task_train_base(**ctx):
    ti = ctx['ti']
    X_train_fe = pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_train_fe_path"))
    y_train    = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_train_path"))
    train_base_models(X_train_fe, y_train, models_dir=MODELS_DIR)

def task_tune(**ctx):
    ti = ctx['ti']
    X_train_fe = pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_train_fe_path"))
    y_train    = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_train_path"))
    best_estimators = tune_models(X_train_fe, y_train)
    # pickling the dict of estimators:
    pd.to_pickle(best_estimators, f"{MODELS_DIR}/best_estimators.pkl")
    ti.xcom_push(key="best_estimators_path", value=f"{MODELS_DIR}/best_estimators.pkl")

def task_build_ensemble(**ctx):
    ti = ctx['ti']
    X_train_fe      = pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_train_fe_path"))
    y_train         = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_train_path"))
    best_estimators = pd.read_pickle(ti.xcom_pull(task_ids="tune_models", key="best_estimators_path"))
    ensemble = build_ensemble(best_estimators, X_train_fe, y_train, models_dir=MODELS_DIR)
    all_models = {**best_estimators, "ensemble": ensemble}
    pd.to_pickle(all_models, f"{MODELS_DIR}/all_models.pkl")
    ti.xcom_push(key="all_models_path", value=f"{MODELS_DIR}/all_models.pkl")

def task_evaluate(**ctx):
    ti = ctx['ti']
    all_models = pd.read_pickle(ti.xcom_pull(task_ids="build_ensemble", key="all_models_path"))
    X_test_fe  = pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_test_fe_path"))
    y_test     = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_test_path"))
    metrics_df = evaluate_models(all_models, X_test_fe, y_test)
    metrics_df.to_pickle(f"{MODELS_DIR}/metrics_df.pkl")
    ti.xcom_push(key="metrics_df_path", value=f"{MODELS_DIR}/metrics_df.pkl")

def task_select_save(**ctx):
    ti         = ctx['ti']
    metrics_df = pd.read_pickle(ti.xcom_pull(task_ids="evaluate", key="metrics_df_path"))
    all_models = pd.read_pickle(ti.xcom_pull(task_ids="build_ensemble", key="all_models_path"))
    best_name, _ = select_and_save_best(
        metrics_df, all_models,
        pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_test_fe_path")),
        pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_test_path")),
        metrics_txt_path=f"{MODELS_DIR}/metrics.txt",
    )
    ti.xcom_push(key="best_name", value=best_name)

def task_plot_target(**ctx):
    raw = pd.read_csv(DATA_PATH)
    raw["placement_status"] = raw["placement_status"].map({"Placed":1,"Not Placed":0})
    plot_target_distribution(raw)

def task_plot_corr(**ctx):
    raw = pd.read_csv(DATA_PATH)
    raw["placement_status"] = raw["placement_status"].map({"Placed":1,"Not Placed":0})
    plot_feature_correlations(raw, ["gpa_or_score","test_score","year_of_enrollment","graduation_year"])

def task_plot_roc(**ctx):
    ti         = ctx['ti']
    all_models = pd.read_pickle(ti.xcom_pull(task_ids="build_ensemble", key="all_models_path"))
    X_test_fe  = pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_test_fe_path"))
    y_test     = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_test_path"))
    plot_roc_curves(all_models, X_test_fe, y_test)

def task_plot_cm(**ctx):
    ti         = ctx['ti']
    best_name  = ti.xcom_pull(task_ids="select_save", key="best_name")
    all_models = pd.read_pickle(ti.xcom_pull(task_ids="build_ensemble", key="all_models_path"))
    X_test_fe  = pd.read_pickle(ti.xcom_pull(task_ids="engineer", key="X_test_fe_path"))
    y_test     = pd.read_pickle(ti.xcom_pull(task_ids="preprocess", key="y_test_path"))
    plot_confusion_matrix(all_models[best_name], X_test_fe, y_test, name=best_name)

with DAG(
    dag_id="hw2_ml_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    t1  = PythonOperator(task_id="preprocess",     python_callable=task_preprocess)
    t2  = PythonOperator(task_id="engineer",       python_callable=task_engineer)
    t3  = PythonOperator(task_id="train_base",     python_callable=task_train_base)
    t4  = PythonOperator(task_id="tune_models",    python_callable=task_tune)
    t5  = PythonOperator(task_id="build_ensemble", python_callable=task_build_ensemble)
    t6  = PythonOperator(task_id="evaluate",       python_callable=task_evaluate)
    t7  = PythonOperator(task_id="select_save",    python_callable=task_select_save)
    t8  = PythonOperator(task_id="plot_target",    python_callable=task_plot_target)
    t9  = PythonOperator(task_id="plot_corr",      python_callable=task_plot_corr)
    t10 = PythonOperator(task_id="plot_roc",       python_callable=task_plot_roc)
    t11 = PythonOperator(task_id="plot_conf_matrix", python_callable=task_plot_cm)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> [t8, t9] >> t10 >> t11
