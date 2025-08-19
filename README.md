
# Foreign Student Placement ML Pipeline

---

## 🏫 Homework 3: MLflow Integration and Model Drift Detection

### Project Overview

Building on the core ML pipeline from Homework 1 and the containerized deployment from Homework 2, this iteration introduces MLflow for experiment tracking and model registry management, along with Evidently AI for comprehensive data drift detection. The enhanced pipeline now provides production-ready monitoring capabilities that automatically detect when models require retraining due to data distribution changes.

As someone who once had the challenges of studying abroad myself, I know firsthand how difficult it can be to secure an internship or job after graduation. International students often face visa restrictions, language barriers, and have limited local networks that can leave even the most qualified candidates at a disadvantage.

This small project aims to gather insights and potentially level the playing field by predicting placement success using a global student migration dataset, so universities and career services can intervene earlier and support those who need it most.

By training and comparing models such as Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN, we not only identify which algorithms perform best but also figure out the most critical factors influencing placement.

---

### How to Get the Data

I include a small sample in `data/` for quick testing. To download the full dataset:

1. Go to  
   [Kaggle Dataset](https://www.kaggle.com/datasets/atharvasoundankar/global-student-migration-and-higher-education-trends)  
2. Download the CSV and save it as  
   `data/global_student_migration.csv`

---

### Setup Instructions

```bash
git clone https://github.com/aim-msds-pt-2025b/4de89279943ea0e42ef94b23123347ac714c2e6b47774d13be052f8c7e4031ab_foreign_student_placement.git
cd 4de89279943ea0e42ef94b23123347ac714c2e6b47774d13be052f8c7e4031ab_foreign_student_placement
git checkout hw1-snapshot

# initialize the venv and install all runtime + dev deps
uv init --dev

# if you ever need to re-sync (e.g. after adding a new dependency):
uv sync --dev

# finally, install your pre-commit hooks
pre-commit install
````

> **Optional backup**:
>
> ```bash
> pip install -r requirements.txt
> ```

---

### Folder Structure

```
.
├── data/                        # raw & sample CSV
├── notebooks/                   # exploratory & final notebooks
├── src/                         # modular pipeline code
├── models/                      # saved .pkl artifacts
├── reports/                     # metrics, confusion matrices, JSON/CSV
│   └── figures/                 # PNGs by visualization.py
├── tests/                       # pytest unit tests
├── .pre-commit-config.yaml      # Black + Ruff hooks
├── requirements.txt             # backup runtime deps
├── pyproject.toml               # project metadata & deps
└── README.md
```

---

### Pre-commit Configuration

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        args: [--line-length=88]

  - repo: https://github.com/charliermarsh/ruff
    rev: main
    hooks:
      - id: ruff
        args: [--fix]
```

---

## 🚀 Homework 2: Containerization & Orchestration

### Project Overview

Building on my core ML pipeline, I now containerize everything with Docker for environment consistency and orchestrate the end-to-end workflow in Apache Airflow, gaining clear DAG definitions, retries, logging, and a UI for monitoring.

---

### Setup Instructions

1. **Install Docker & Compose**
   Follow official docs:

   * Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
   * Compose: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

2. **Clone & switch branch**

   ```bash
   git clone https://github.com/aim-msds-pt-2025b/4de89279943ea0e42ef94b23123347ac714c2e6b47774d13be052f8c7e4031ab_foreign_student_placement.git
   cd 4de89279943ea0e42ef94b23123347ac714c2e6b47774d13be052f8c7e4031ab_foreign_student_placement
   git checkout hw2-docker-airflow
   ```

3. **Start Airflow stack**

   ```bash
   cd deploy/airflow
   docker-compose build
   docker-compose up -d
   ```

4. **Initialize & create admin user**

   ```bash
   docker-compose exec webserver airflow db init
   docker-compose exec webserver \
     airflow users create \
       --username admin \
       --firstname Admin \
       --lastname User \
       --role Admin \
       --email admin@example.com \
       --password airflow
   ```

5. **Open UI & trigger DAG**
   [http://127.0.0.1:8080](http://127.0.0.1:8080) → log in `admin`/`airflow` → trigger **hw2\_ml\_pipeline**.

---
### Docker Integration

#### Dockerfile
My Dockerfile is based on the official `python:3.12-slim` image to keep the build lightweight while providing full Python support. I copy my ML pipeline code (`src/`) and the Airflow deployment files (`deploy/airflow/`) into the container. Inside the container, I install all Python dependencies listed in `pyproject.toml`, ensuring that the exact versions I tested locally are reproduced in every build. Finally, I set the container’s entrypoint to launch Airflow (either the webserver or the scheduler), so that when the container starts it automatically initializes the Airflow component without any additional commands.

#### Building the Image
To build my custom Airflow image, I run:

```bash
docker-compose build
```

This reads the `docker-compose.yml` which references my Dockerfile, pulls the base Python image, installs all dependencies, and packages my code into a ready-to-run Airflow container. Because the build context includes my `pyproject.toml`, any change to dependencies will trigger a rebuild of the environment layer.

**Running Containers**
I use `docker-compose up -d` to spin up three services:

* **Postgres**: Serves as Airflow’s metadata database.
* **Webserver**: Runs Airflow’s web UI on port 8080.
* **Scheduler**: Executes the DAG and schedules tasks.

By defining the entrypoint in each service, Docker Compose automatically starts the correct Airflow component—no manual commands inside the container are needed.

**Volume Mounting Strategy**
To achieve reproducibility and allow live code changes, I mount host directories into the containers:

1. **`./dags` → `/opt/airflow/dags`**
   All DAG definitions live here. Mounting it means editing a DAG file on the host immediately reflects in the Airflow UI without rebuilding the image.

2. **`./logs` → `/opt/airflow/logs`**
   Task logs and scheduler logs are persisted to the host, so they survive container restarts and can be inspected directly from the file system.

3. **`../../src` → `/app/src`**
   My core ML pipeline modules (preprocessing, training, etc.) are mounted into the Airflow container’s Python path. This ensures Airflow’s PythonOperator can import and run them as if they were installed in the container.

4. **`../../data` → `/app/data`**
   The raw CSV dataset is made available to both DockerOperator (in the pipeline image) and PythonOperator tasks without embedding large data files into the image.

5. **`../../models` → `/app/models`**
   Output model artifacts (pickles, metrics, figures) are written here. By mounting it, the host filesystem always contains the latest trained models and evaluation outputs, making post-run analysis straightforward.

6. **`/var/run/docker.sock` → `/var/run/docker.sock`**
   This mount allows the Airflow `DockerOperator` to spin up additional containers (e.g., the ML pipeline image) from within the Airflow container itself.

Together, these mounts decouple the containerized runtime environment from the host’s file structure while providing seamless two-way synchronization of code, data, logs, and artifacts—key to a reproducible, editable, and inspectable MLOps setup.


---

### Deploy Folder Structure

```
deploy/
└── airflow/
    ├── dags/
    │   └── ml_pipeline_dag.py     # Airflow DAG definition
    ├── logs/                      # persisted task logs
    ├── Dockerfile.airflow        # custom Airflow image build
    └── docker-compose.yml        # local Airflow stack
└── docker/
    ├── Dockerfile                # ML pipeline image build
    └── .dockerignore
```

---

### Airflow DAG

* **File**: `deploy/airflow/dags/ml_pipeline_dag.py`

* **Tasks** using `PythonOperator`:

  1. **preprocess**
  2. **engineer**
  3. **train\_base**
  4. **tune\_models**
  5. **build\_ensemble**
  6. **evaluate**
  7. **select\_save**
  8. **plot\_target** & **plot\_corr** (parallel)
  9. **plot\_roc** → **plot\_conf\_matrix**

* **Dependency graph**:
  `preprocess → engineer → train_base → tune_models → build_ensemble → evaluate → select_save → [plot_target, plot_corr] → plot_roc → plot_conf_matrix`

* **Scheduling**: manual only (`schedule_interval=None`, `catchup=False`).

---

### Reflection  
On my home computer Docker would simply refuse to start until I went into the BIOS and enabled hardware virtualization—an extra step I never needed on my work laptop or personal laptop. This BIOS tweak taught me that, beyond code and containers, underlying hardware settings can make or break your MLOps setup, and now I always check that virtualization flag first which I found in the task manager.

I also began by wrapping the entire ML pipeline in one big `DockerOperator` task, but found the Airflow UI much more insightful when each stage—preprocessing, feature engineering, model training, tuning, evaluation, plotting—was its own `PythonOperator`. Splitting tasks this way made dependencies explicit, improved retry granularity, and gave me a clearer picture of where things might fail or need tuning.

Juggling these challenges across three different machines and adapting my DAG design deepened my appreciation for immutable, reproducible environments and for Airflow’s orchestration power. Wrestling with hardware settings, volume mounts, and operator choices was frustrating at times, but now I’m confident I can deploy and debug a robust MLOps workflow anywhere.  

---

## 🧪 Homework 3: Verification, Drift, and MLflow Registry

This section documents the additional HW3 requirements: MLflow experiment tracking and model registration, Evidently-based drift detection with a drift report, exact setup/verification commands, and submission details.

### Setup Instructions (Docker + Services)

```bash
# From repo root
docker compose -f deploy/airflow/docker-compose.yml up -d

# Verify UIs
curl http://localhost:8080  # Airflow UI should return HTML
curl http://localhost:5000  # MLflow UI should return HTML
```

Notes:
- MLflow runs on http://localhost:5000
- Airflow runs on http://localhost:8080
- MLflow tracking URI is set via environment (MLFLOW_TRACKING_URI) and used in code.

### Test Commands

```bash
# 1) Test standalone pipeline (will evaluate and may register if threshold met)
python src/run_pipeline.py

# 2) Test Airflow DAG end-to-end (one-off dry run)
docker exec airflow-webserver-1 airflow dags test ml_pipeline_dag 2025-08-02

# 3) Verify MLflow UI reachable
curl http://localhost:5000
```

Expected behavior:
- Standalone pipeline: when run with drifted data, it raises a clear error indicating drift and need for retraining. In my Airflow implementation, I instead branch on drift with a `BranchPythonOperator`: if drift is detected the `retrain_model` task runs; otherwise the pipeline completes. This keeps the DAG green while still responding to drift.

### MLflow Integration (tracking + registration)

- Parameters and exactly two evaluation metrics for classification (accuracy, f1_score) are logged per model in `src/evaluation.py`.
- Tracking URI is configured via `MLFLOW_TRACKING_URI` (defaults to the MLflow service). The pipeline also supports a host default of `http://localhost:5000`.
- After evaluation, the best model’s accuracy is compared to a threshold (default `0.8`, overridable with env var `ML_THRESHOLD`). If met, the model is logged and registered in the MLflow Model Registry. You’ll see runs under the “Default” experiment and any registered models in the Models section of the MLflow UI.

### Model Drift Detection (simulation and detection)

I generate a “drifted” test set during preprocessing and compare it with the original test set. Drift detection is implemented in `src/drift_detection.py` using Evidently’s `Report` with drift metrics (e.g., `DriftedColumnsCount`, `ValueDrift`) to ensure tooling integration. Because Evidently’s programmatic extraction can be verbose, I also compute statistical drift (KS test for numeric features, chi-square for categorical) as a robust fallback. Results are saved to `reports/drift_report.json` with required keys: `drift_detected`, `feature_drifts` (per-feature scores), and `overall_drift_score`. In Airflow, the DAG reads this report and branches to `retrain_model` if drift is detected. This design prioritizes automation (retraining path) without failing the entire DAG while still surfacing drift evidence for operators.

### Folder Structure (new/updated, rationale)

I added/updated:
- `deploy/airflow/dags/ml_pipeline_dag.py`: DAG includes a `drift_detection` task and a `branch_on_drift` decision that routes to `retrain_model` if drift is found. This keeps orchestration explicit and recoverable.
- `src/drift_detection.py`: Encapsulates Evidently-based detection with a statistical fallback. Centralizing this logic makes unit testing and reuse simpler.
- `reports/drift_report.json`: Persisted output for downstream branching and auditability. Keeping it in `reports/` co-locates artifacts with other evaluation outputs.
- MLflow usage across `src/` to log metrics/params and optionally register the best model when thresholds are met. This ties experimentation with deployment governance.

These additions create a coherent loop: evaluate → detect drift → retrain if needed → track and register the improved model, all observable in Airflow and MLflow UIs.

### Verification Checklist

- MLflow UI loads at http://localhost:5000 and shows runs for the “Default” experiment.
- `reports/drift_report.json` exists and contains: `drift_detected`, `feature_drifts`, `overall_drift_score`.
- Airflow DAG `ml_pipeline_dag` runs and chooses `retrain_model` when drift is detected.
- Best model accuracy is compared to threshold; if met, model is registered (visible in MLflow “Models”).
- README reflects the HW3 sections and commands; branch name is `hw3-mlflow-drift`.


### Thresholds and Justification

Classification threshold defaults to `accuracy > 0.8`. If this is unrealistic for a given dataset or split, override via `ML_THRESHOLD` or justify an alternative threshold in this README (e.g., F1 focus for imbalanced data). For regression and clustering tasks, analogous thresholds are: `MSE < 0.1`, `silhouette_score > 0.5`; if not applicable, provide a short rationale and chosen alternatives.

### Testing Instructions

I verify functionality with these steps:

```bash
# 0) Start services (first terminal)
docker compose -f deploy/airflow/docker-compose.yml up -d

# 1) Standalone pipeline (creates runs, metrics, reports)
python src/run_pipeline.py

# Optional: force model registration demo by lowering threshold
# (Windows PowerShell)  $env:ML_THRESHOLD=0.65; python src/run_pipeline.py
# (Linux/macOS Git Bash) ML_THRESHOLD=0.65 python src/run_pipeline.py

# 2) Airflow DAG dry run (tests tasks without scheduling)
docker exec airflow-webserver-1 airflow dags test ml_pipeline_dag 2025-08-02

# 3) Spot-check drift artifact
cat reports/drift_report.json

# 4) Verify UIs are reachable
curl http://localhost:8080  # Airflow UI HTML
curl http://localhost:5000  # MLflow UI HTML
```

What to expect:
- `reports/evaluation_results.json` and `reports/drift_report.json` are created.
- MLflow shows new runs under the Default experiment; models may register if the threshold is met.
- In Airflow, drift detection triggers the `retrain_model` branch when drift is found.

### Reflection (HW3)

Working on MLflow and drift integration brought several production-like challenges. At first, I faced permission errors when saving model artifacts, which caused runs to fail. To stabilize things, I limited logging to parameters and metrics and ensured the tracking server URI was consistent across all modules; this fixed the issue where the MLflow UI looked empty because some scripts used file:./mlruns while others used the server. Drift detection was another hurdle since Evidently’s results were not straightforward to extract, so I combined its reports with basic statistical tests (KS/chi-square) to reliably produce a compact JSON summary. In Airflow, I realized it was better to branch on drift rather than fail the whole DAG, so I used a BranchPythonOperator to trigger retraining only when drift was detected. Testing DAGs also required attention, as airflow dags test failed on future dates, so I used today’s date for task tests and fixed past dates for end-to-end runs. On the container side, I learned that using pip with a pinned requirements.txt was more reliable than uv inside the Airflow image, so I stuck with that to keep builds stable. Finally, model registration didn’t trigger at first because my accuracy was below the threshold, so I exposed ML_THRESHOLD as a parameter to test the flow before restoring the stricter default.

