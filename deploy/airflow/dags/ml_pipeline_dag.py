# deploy/airflow/dags/ml_pipeline_dag.py

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime

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
        command="",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            # Mount the named volume `hw2_data` into /app/data inside the pipeline container:
            Mount(source="hw2_data", target="/app/data", type="volume"),
            Mount(source="hw2_models", target="/app/models", type="volume"),
        ],
    )
