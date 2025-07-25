# Foreign Student Placement ML Pipeline

---

## Project Overview

As someone who once had the challenges of studying abroad myself, I know firsthand how difficult it can be to secure an internship or job after graduation. International students often face visa restrictions, language barriers, and have limited local networks that can leave even the most qualified candidates at a disadvantage. 

This small project aims to gather insights and potentially level the playing field by predicting placement success using a global student migration dataset, so universities and career services can intervene earlier and support those who need it most.

By training and comparing models such as Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN, we not only identify which algorithms perform best but also figure out the most critical factors influencing placement. In today’s global job market, demand for skilled graduates often exceeds supply. Using data-driven insights helps institutions make better use of their resources and support student success worldwide.

---

## How to Get the Data

We include a small sample in `data/` for quick testing. To download the full dataset:

1. Go to  
   [Kaggle Dataset](https://www.kaggle.com/datasets/atharvasoundankar/global-student-migration-and-higher-education-trends)
2. Download the CSV and save it as  
   `data/global_student_migration.csv`

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/aim-msds-pt-2025b/748a75ce7964f4331a0d0f4ee45adabd8bb41932fbb0ca6ec6b08004e4a7cbf9_foreign_student_placement.git
cd foreign_student_placement
```

### 2. Create & activate your UV environment

```bash
python3.12 -m venv uv
source uv/bin/activate      # macOS/Linux
uv\Scripts\activate         # Windows
```

### 3. Install dependencies using `uv` and `pyproject.toml`

```bash
uv pip install -e .[dev]
```

> 💡 If `uv` is not installed, install it first via:
>
> ```bash
> pip install uv
> ```

### 4. (Optional) Install from `requirements.txt` (backup method)

If you prefer or need to install from a frozen requirements file:

```bash
pip install -r requirements.txt
```

### 5. Install pre-commit hooks (for formatting and linting)

```bash
pre-commit install
```

---

## Folder Structure

```
.
├── data/                        # raw & minimal sample CSV
├── notebooks/                   # exploratory & final “hash” notebooks
├── src/                         # modular pipeline code
├── models/                      # saved .pkl model artifacts
├── reports/                     # metrics, confusion matrices, JSON/CSV summaries
│   └── figures/                 # PNGs produced by visualization.py
├── tests/                       # pytest unit tests
├── .pre-commit-config.yaml      # Black + Ruff hooks
├── requirements.txt             # backup for runtime deps
├── pyproject.toml               # project metadata & dependency config
└── README.md
```

---

## Code Sections

All core logic lives under `src/`, with one file per pipeline stage:

- **`data_preprocessing.py`**  
  Loads the CSV, drops identifier/leakage columns, maps “Placed”/“Not Placed” to 1/0, removes rows missing core features, splits into train/test (with stratification), and scales numeric columns.

- **`feature_engineering.py`**  
  Builds new features: study-duration interactions, polynomial terms, quantile buckets, count-encoding, category combinations, K-means cluster labels, then one-hot encodes & aligns train/test.

- **`model_training.py`**  
  Defines, trains & saves multiple base classifiers (RandomForest, GBM, LogisticRegression, SVM, kNN), runs hyperparameter search (RandomizedSearchCV), and assembles a soft-voting ensemble.

- **`run_pipeline.py`**  
  Orchestrates the full workflow: preprocess → feature engineer → train & tune models → evaluate → visualize → export final metrics.

---

## Optional: Visualization

- **`visualization.py`**  
  Generates and exports key EDA and performance plots (target distributions, correlation heatmaps, ROC curves, confusion matrices) into `reports/figures/`.

---

## Pre-commit Configuration

We enforce formatting and linting on each commit using:

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

- **Black** ensures consistent code formatting.
- **Ruff** identifies and auto-fixes common linting issues.

---

## Reflection

When I first started this project, setting up a clean development environment and code-quality tooling proved more challenging than I expected. I lost my `.pre-commit-config.yaml` file at one point and had to rebuild it from scratch—reinstalling Black and Ruff, re-running `pre-commit install`, and even rewriting parts of my Git history to restore the hooks. I also ran into Git rebase conflicts and “non-fast-forward” errors when pushing to the remote repository, which forced me to learn how to safely abort or continue a rebase without losing work.

On the data side, my simple pytest fixtures kept collapsing the entire mini-datasets because the string `"None"` in some columns was automatically interpreted as a missing value (`NaN`). That caused every toy CSV I created to be dropped entirely by our `dropna()` step. To address this, I explored Pandas’ CSV-reading options (`keep_default_na=False`) and explicitly mapped `"None"` to a valid category so that only truly missing values would be removed. I also encountered scikit-learn errors when splitting very small datasets—using an integer `test_size` versus a fractional `test_size` required special handling to prevent empty train or test splits.

With the data loading and testing pipeline finally stable, I was able to focus on feature engineering—creating interaction and ratio features, clustering, bucketization, and count-encoding—and then on model training and hyperparameter tuning. I compared five algorithms—Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN—and combined them in a soft-voting ensemble to boost overall performance. I then added a dedicated Visualization module so that all exploratory plots and model performance figures can be generated automatically.

Despite the many bumps along the way—lost configs, rebasing nightmares, test failures, and convergence warnings—the end result is a fully automated, reproducible pipeline that enforces code style, validates data loading, trains and tunes models, generates plots, and saves final metrics. This journey reinforced for me the value of automation, thorough testing, and clear error handling at every stage of a machine learning project.
