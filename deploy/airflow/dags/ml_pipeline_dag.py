# deploy/airflow/dags/ml_pipeline_dag.py
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
import os

# >> NEW: import Mount
from docker.types import Mount

# Resolve host paths relative to this file
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
HOST_DATA = os.path.join(ROOT, "data")
HOST_MODELS = os.path.join(ROOT, "models")

with DAG(
    dag_id="hw2_ml_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_pipeline = DockerOperator(
        task_id="run_full_pipeline",
        image="4de89279943ea0e42ef94b23123347ac714c2e6b47774d13be052f8c7e4031ab-ml-pipeline",
        api_version="auto",
        auto_remove=True,
        command="",  # uses the image’s ENTRYPOINT
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        # <<-- use mounts with docker.types.Mount (not volumes or raw strings) -->>
        mounts=[
            Mount(source=HOST_DATA, target="/app/data", type="bind"),
            Mount(source=HOST_MODELS, target="/app/models", type="bind"),
        ],
    )
