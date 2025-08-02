
# Foreign Student Placement ML Pipeline

---

## 🏫 Homework 1: Core ML Pipeline

### Project Overview

As someone who once had the challenges of studying abroad myself, I know firsthand how difficult it can be to secure an internship or job after graduation. International students often face visa restrictions, language barriers, and have limited local networks that can leave even the most qualified candidates at a disadvantage.

This small project aims to gather insights and potentially level the playing field by predicting placement success using a global student migration dataset, so universities and career services can intervene earlier and support those who need it most.

By training and comparing models such as Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN, we not only identify which algorithms perform best but also figure out the most critical factors influencing placement.

---

### How to Get the Data

We include a small sample in `data/` for quick testing. To download the full dataset:

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

Building on our core ML pipeline, we now containerize everything with Docker for environment consistency and orchestrate the end-to-end workflow in Apache Airflow, gaining clear DAG definitions, retries, logging, and a UI for monitoring.

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
Our Dockerfile is based on the official `python:3.12-slim` image to keep the build lightweight while providing full Python support. We copy our ML pipeline code (`src/`) and the Airflow deployment files (`deploy/airflow/`) into the container. Inside the container, we install all Python dependencies listed in `pyproject.toml`, ensuring that the exact versions we tested locally are reproduced in every build. Finally, we set the container’s entrypoint to launch Airflow (either the webserver or the scheduler), so that when the container starts it automatically initializes the Airflow component without any additional commands.

#### Building the Image
To build our custom Airflow image, we run:

```bash
docker-compose build
```

This reads the `docker-compose.yml` which references our Dockerfile, pulls the base Python image, installs all dependencies, and packages our code into a ready-to-run Airflow container. Because the build context includes our `pyproject.toml`, any change to dependencies will trigger a rebuild of the environment layer.

**Running Containers**
We use `docker-compose up -d` to spin up three services:

* **Postgres**: Serves as Airflow’s metadata database.
* **Webserver**: Runs Airflow’s web UI on port 8080.
* **Scheduler**: Executes the DAG and schedules tasks.

By defining the entrypoint in each service, Docker Compose automatically starts the correct Airflow component—no manual commands inside the container are needed.

**Volume Mounting Strategy**
To achieve reproducibility and allow live code changes, we mount host directories into the containers:

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
