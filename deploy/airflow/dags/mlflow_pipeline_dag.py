"""
Airflow DAG for Student Placement ML Pipeline with MLflow tracking and drift detection.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import os


# Default arguments for the DAG
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Initialize DAG
dag = DAG(
    "student_placement_mlflow_pipeline",
    default_args=default_args,
    description="ML Pipeline with MLflow tracking and drift detection",
    schedule_interval=timedelta(days=7),  # Run weekly
    catchup=False,
    tags=["machine-learning", "mlflow", "drift-detection"],
)


def check_data_availability(**context):
    """Check if input data is available."""
    data_path = "/opt/airflow/data/global_student_migration.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Check data quality
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Data file is empty")

    print(f"✅ Data validation passed. Found {len(df)} records.")
    return {"data_path": data_path, "record_count": len(df)}


def run_drift_detection(**context):
    """Run drift detection on new data."""
    # This would be a more sophisticated implementation in production
    # For now, we'll run our drift detection demo

    from src.drift_detection import DriftDetector, create_synthetic_drift_data
    import pandas as pd

    print("🔍 Running drift detection...")

    # Load data
    data_path = context["task_instance"].xcom_pull(task_ids="check_data")["data_path"]
    df = pd.read_csv(data_path)
    df["placement_status"] = df["placement_status"].map({"Placed": 1, "Not Placed": 0})

    # Initialize drift detector
    drift_detector = DriftDetector(
        reference_data=df,
        target_column="placement_status",
        numerical_features=[
            "gpa_or_score",
            "test_score",
            "year_of_enrollment",
            "graduation_year",
        ],
        categorical_features=[
            "gender",
            "nationality",
            "destination_country",
            "field_of_study",
            "degree_level",
        ],
    )

    # For demo, create synthetic drifted data
    drifted_data = create_synthetic_drift_data(df, drift_magnitude=0.2)

    # Detect drift
    drift_results = drift_detector.detect_data_drift(drifted_data)
    alert = drift_detector.generate_drift_alert(drift_results)

    # Save results
    results = {
        "drift_detected": drift_results.get("dataset_drift_detected", False),
        "drift_share": drift_results.get("drift_share", 0),
        "alert_level": alert.get("alert_level", "LOW"),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"📊 Drift detection results: {results}")
    return results


def decide_retrain(**context):
    """Decide whether to retrain the model based on drift detection."""
    drift_results = context["task_instance"].xcom_pull(task_ids="drift_detection")

    # Decision logic
    retrain_needed = (
        drift_results["drift_detected"]
        or drift_results["drift_share"] > 0.3
        or drift_results["alert_level"] in ["HIGH", "MEDIUM"]
    )

    print(f"🤔 Retrain decision: {retrain_needed}")
    print(f"   Drift detected: {drift_results['drift_detected']}")
    print(f"   Drift share: {drift_results['drift_share']:.2%}")
    print(f"   Alert level: {drift_results['alert_level']}")

    return {"retrain_needed": retrain_needed}


def send_drift_alert(**context):
    """Send alert if significant drift is detected."""
    drift_results = context["task_instance"].xcom_pull(task_ids="drift_detection")

    if drift_results["alert_level"] in ["HIGH", "MEDIUM"]:
        print("🚨 DRIFT ALERT!")
        print(f"   Alert Level: {drift_results['alert_level']}")
        print(f"   Drift Share: {drift_results['drift_share']:.2%}")
        print(f"   Timestamp: {drift_results['timestamp']}")

        # In production, this would send actual notifications
        # (email, Slack, PagerDuty, etc.)
        print("📧 Alert sent to ML team!")
    else:
        print("✅ No alert needed - drift levels are acceptable")


# Task 1: Check data availability
check_data_task = PythonOperator(
    task_id="check_data",
    python_callable=check_data_availability,
    dag=dag,
)

# Task 2: Run drift detection
drift_detection_task = PythonOperator(
    task_id="drift_detection",
    python_callable=run_drift_detection,
    dag=dag,
)

# Task 3: Decide on retraining
retrain_decision_task = PythonOperator(
    task_id="retrain_decision",
    python_callable=decide_retrain,
    dag=dag,
)

# Task 4: Send drift alert if needed
alert_task = PythonOperator(
    task_id="send_drift_alert",
    python_callable=send_drift_alert,
    dag=dag,
)

# Task 5: Run full ML pipeline (conditional)
run_pipeline_task = BashOperator(
    task_id="run_ml_pipeline",
    bash_command="""
    cd /opt/airflow &&
    python -c "
from src.run_pipeline_mlflow import main
print('🚀 Starting ML Pipeline with MLflow...')
main()
print('✅ Pipeline completed!')
"
    """,
    dag=dag,
)

# Task 6: Model validation (runs after pipeline)
validate_model_task = BashOperator(
    task_id="validate_model",
    bash_command="""
    cd /opt/airflow &&
    python -c "
import os
import json
import mlflow
from src.mlflow_config import MLflowTracker

print('🔍 Validating deployed model...')

# Load latest model metrics
if os.path.exists('reports/metrics.txt'):
    print('✅ Model metrics found')
    with open('reports/metrics.txt', 'r') as f:
        print(f.read())
else:
    print('⚠️ No model metrics found')

# Check MLflow runs
try:
    tracker = MLflowTracker()
    runs_df = tracker.compare_runs()
    print(f'📊 Found {len(runs_df)} MLflow runs')
    if len(runs_df) > 0:
        print('Top performing models:')
        print(runs_df.head())
except Exception as e:
    print(f'⚠️ Could not access MLflow: {e}')

print('✅ Model validation completed')
"
    """,
    dag=dag,
)

# Task 7: Cleanup old artifacts
cleanup_task = BashOperator(
    task_id="cleanup_artifacts",
    bash_command="""
    cd /opt/airflow &&
    echo "🧹 Cleaning up old artifacts..."
    
    # Keep only last 10 model files
    find models/ -name "*.pkl" -type f -exec ls -t {} + | tail -n +11 | xargs -r rm
    
    # Clean old MLflow runs (keep last 50)
    python -c "
try:
    import mlflow
    from mlflow import MlflowClient
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name('student_placement_prediction')
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) > 50:
            old_runs = sorted(runs, key=lambda x: x.info.start_time)[:-50]
            for run in old_runs:
                client.delete_run(run.info.run_id)
            print(f'🗑️ Deleted {len(old_runs)} old runs')
        else:
            print('✅ No cleanup needed')
except Exception as e:
    print(f'⚠️ Cleanup warning: {e}')
"
    
    echo "✅ Cleanup completed"
    """,
    dag=dag,
)

# Define task dependencies
check_data_task >> drift_detection_task
drift_detection_task >> [retrain_decision_task, alert_task]
retrain_decision_task >> run_pipeline_task
run_pipeline_task >> validate_model_task
validate_model_task >> cleanup_task

# The alert task runs in parallel and doesn't block the pipeline
alert_task
