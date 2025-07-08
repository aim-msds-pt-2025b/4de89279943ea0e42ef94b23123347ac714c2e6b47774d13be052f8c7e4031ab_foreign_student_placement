# Foreign Student Placement ML Pipeline

## Project Overview

This project tackles the problem of predicting whether a student will secure a placement (internship/job) after graduation, based on academic performance, background, and visa status. We leverage a global student migration dataset to train and compare several models—Random Forests, Gradient Boosting, Logistic Regression, SVM, KNN, and a soft‐voting ensemble—to identify the best predictor of placement success.

## How to Get the Data

The primary dataset is set at `data/global_student_migration.csv` which was taken from Kaggle.
If you need to download it yourself:

1. Visit `https://www.kaggle.com/datasets/atharvasoundankar/global-student-migration-and-higher-education-trends`.
2. Download the CSV and save it as `data/global_student_migration.csv`.

## Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/foreign_student_placement.git
   cd foreign_student_placement
   ```

2. **Create & activate UV environment**

   ```bash
   python3.12 -m venv uv
   source uv/bin/activate      # macOS/Linux
   uv\Scripts\activate         # Windows
   ```

3. **Install runtime requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install dev tools & hooks**

   ```bash
   pip install pre-commit black ruff pytest
   pre-commit install
   ```

5. **Regenerate requirements** *(after adding new dependencies)*

   ```bash
   pip freeze > requirements.txt
   ```

## Folder Structure

```
.
├── data/                        # raw & minimal sample data (CSV)
├── notebooks/                   # exploratory & hash notebooks
├── src/                         # source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── run_pipeline.py
├── models/                      # serialized .pkl model files
├── reports/                     # metrics, confusion matrices, JSON summaries
├── tests/                       # pytest unit tests
├── .pre-commit-config.yaml      # pre-commit hooks
├── requirements.txt             # pinned dependencies
└── README.md
```

## Pre-commit Configuration

We enforce formatting and linting on every commit:

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

## Reflection

One challenge was dealing with `"None"` strings in the CSV being interpreted as `NaN`, causing our tiny pytest toy-datasets to drop all rows. We resolved this by reading CSVs with `keep_default_na=False` (or explicitly mapping `"None"` to a valid category) so that only truly missing values are removed by `dropna()`.

---

*You can optionally add sections such as “Running Visualizations”, “Troubleshooting”, or “Model Deployment Steps” as needed.*
