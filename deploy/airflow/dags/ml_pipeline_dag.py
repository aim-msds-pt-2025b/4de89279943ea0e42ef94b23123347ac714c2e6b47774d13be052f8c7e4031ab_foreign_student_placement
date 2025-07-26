from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# Let Airflow find your ML code
sys.path.append("/opt/airflow/src")

from run_pipeline import main as run_pipeline

with DAG(
    dag_id="hw2_ml_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    run_all = PythonOperator(
        task_id="run_full_pipeline",
        python_callable=run_pipeline,
    )
