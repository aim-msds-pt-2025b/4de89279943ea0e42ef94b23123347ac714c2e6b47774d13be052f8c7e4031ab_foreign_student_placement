````markdown
# Foreign Student Placement ML Pipeline

## Project Overview

## Project Overview

As someone who once had the challenges of studying abroad myself, I know firsthand know how difficult it can be to secure an internship or job after graduation. International students often face visa restrictions, language barriers, and have limited local networks that can leave even the most qualified candidates at a disadvantage. This small project aims to gather insights and potentially level the playing field by predicting placement success using a global student migration dataset, so universities and career services can intervene earlier and support those who need it most.

By training and comparing models such as Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN, we not only identify which algorithms perform best but also figure out the most critical factors influencing placement. In today’s global job market, demand for skilled graduates often exceeds supply. Using data-driven insights helps institutions make better use of their resources and support student success worldwide.
---

## How to Get the Data

We include a small sample in `data/` for quick testing. To download the full dataset:

1. Go to  
   https://www.kaggle.com/datasets/atharvasoundankar/global-student-migration-and-higher-education-trends  
2. Download the CSV and save it as  
   `data/global_student_migration.csv`

---

## Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/aim-msds-pt-2025b/748a75ce7964f4331a0d0f4ee45adabd8bb41932fbb0ca6ec6b08004e4a7cbf9_foreign_student_placement.git
   cd foreign_student_placement
````

2. **Create & activate your UV environment**

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

5. **Regenerate requirements** *(after adding new deps)*

   ```bash
   pip freeze > requirements.txt
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
├── requirements.txt             # pinned runtime deps
├── pyproject.toml               # project metadata & dev extras (pytest, etc.)
└── README.md
```

---

## Code Sections

All core logic lives under `src/`, with one file per pipeline stage:

* **data\_preprocessing.py**
  Loads the CSV, drops identifier/leakage columns, maps “Placed”/“Not Placed” to 1/0, removes rows missing core features, splits into train/Test (with stratification), and scales numeric columns.

* **feature\_engineering.py**
  Builds new features: study‐duration interactions, polynomial terms, quantile buckets, count‐encoding, category combinations, K-means cluster labels, then one-hot encodes & aligns train/test.

* **model\_training.py**
  Defines, trains & saves multiple base classifiers (RandomForest, GBM, LogisticRegression, SVM, kNN), runs hyperparameter search (RandomizedSearchCV), and assembles a soft‐voting ensemble of the tuned models.

* **run\_pipeline.py**
  Orchestrates the entire workflow: preprocess → feature engineer → train & tune models → evaluate → visualize → write out final metrics.

---

# Optional Sections

An optional visualization is added that also lives under `src`.

* **visualization.py**
  Generates and exports key EDA and performance plots (target distributions, correlation heatmaps, ROC curves, confusion matrices) into `reports/figures/`.

---

## Pre-commit Configuration

We enforce formatting and linting on each commit:

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

* **Black** keeps code style consistent.
* **Ruff** finds & auto-fixes lint issues (unused imports, style violations).

---

## Reflection

One challenge was that the raw CSV used the string `"None"` for missing visa fields, which pandas’ default `read_csv` treats as `NaN`. Our tiny pytest datasets then dropped **all** rows in `dropna()`. We fixed this by passing `keep_default_na=False` and/or explicitly mapping `"None"` to a valid category, ensuring only truly missing data is removed.

---


