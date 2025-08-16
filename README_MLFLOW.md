# MLflow and Drift Detection Integration

This project demonstrates the integration of MLflow experiment tracking and data drift detection into an existing machine learning pipeline for student placement prediction.

## 🎯 Project Overview

**Original Pipeline**: A comprehensive ML pipeline for predicting student placement outcomes using multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression, KNN, and Ensemble).

**Enhanced Features**:
- **MLflow Experiment Tracking**: Complete experiment tracking with metrics, parameters, and model versioning
- **Data Drift Detection**: Statistical drift detection using KS tests and Chi-square tests
- **Model Registry**: Automated model registration and versioning
- **Alerting System**: Drift-based alerting and recommendations
- **Airflow Integration**: Automated workflow orchestration

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -e .

# Install MLflow and drift detection dependencies
pip install mlflow>=2.8.0 scipy
```

### 2. Run Enhanced Pipeline

```bash
# Run pipeline with MLflow tracking
python src/run_pipeline_mlflow.py
```

### 3. View MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlruns

# Or use the provided script
./scripts/start_mlflow_ui.sh  # Linux/Mac
./scripts/start_mlflow_ui.bat # Windows
```

## 📊 Features Implemented

### MLflow Integration

#### 1. Experiment Tracking (`src/mlflow_config.py`)
- **Automatic experiment creation**: Creates "student_placement_prediction" experiment
- **Run tracking**: Logs all model training runs with metrics and parameters
- **Model registration**: Automatic model registration to MLflow Model Registry
- **Run comparison**: Generates comparison reports across all experiments

#### 2. Model Metrics Tracked
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Cross-validation scores and standard deviations
- Hyperparameters for each model
- Model artifacts (saved models)

#### 3. Model Registry
- Automatic registration of trained models
- Version management for model iterations
- Stage transitions (Staging → Production)
- Model metadata and lineage tracking

### Drift Detection

#### 1. Statistical Drift Detection (`src/simple_drift_detection.py`)
- **Numerical features**: Kolmogorov-Smirnov test for distribution changes
- **Categorical features**: Chi-square test for category distribution changes
- **Target drift**: Separate monitoring of target variable distribution
- **Configurable thresholds**: Adjustable significance levels and alert thresholds

#### 2. Data Quality Monitoring
- Missing value detection
- Constant column identification
- Duplicate row detection
- Schema consistency checks

#### 3. Alerting System
- **Alert Levels**: LOW, MEDIUM, HIGH based on drift severity
- **Recommendations**: Automated suggestions for handling drift
- **Reporting**: JSON reports with detailed drift information

### Enhanced Pipeline (`src/run_pipeline_mlflow.py`)

#### 1. Integrated Workflow
```
Data Loading → Feature Engineering → Model Training (with MLflow) → 
Evaluation → Best Model Selection → Model Registration → 
Drift Detection Demo → Visualization → Report Generation
```

#### 2. Error Handling
- Graceful degradation when MLflow is unavailable
- Comprehensive error logging and warnings
- Fallback mechanisms for drift detection

#### 3. Reporting
- MLflow run comparison CSV
- Drift detection JSON reports
- Traditional metrics and visualizations

## 🔧 Architecture

### Core Components

1. **MLflowTracker** (`mlflow_config.py`)
   - Centralized MLflow operations
   - Experiment and run management
   - Model registration utilities

2. **SimpleDriftDetector** (`simple_drift_detection.py`)
   - Statistical drift detection
   - Data quality assessment
   - Alert generation

3. **Enhanced Pipeline** (`run_pipeline_mlflow.py`)
   - Orchestrates all components
   - Provides comprehensive logging
   - Handles errors gracefully

### Airflow Integration (`deploy/airflow/dags/mlflow_pipeline_dag.py`)

```
Data Check → Drift Detection → Retrain Decision → Pipeline Execution → 
Model Validation → Cleanup → Alert Notifications
```

**Key Features**:
- Automated drift monitoring
- Conditional retraining based on drift levels
- Model validation and cleanup
- Integrated alerting system

## 📈 Usage Examples

### 1. Basic Pipeline Execution

```python
from src.run_pipeline_mlflow import main
main()  # Runs complete pipeline with MLflow tracking
```

### 2. Drift Detection Only

```python
from src.simple_drift_detection import DriftDetector, create_synthetic_drift_data
import pandas as pd

# Load data
data = pd.read_csv("data/global_student_migration.csv")

# Initialize detector
detector = DriftDetector(
    reference_data=data,
    target_column="placement_status",
    numerical_features=["gpa_or_score", "test_score"],
    categorical_features=["gender", "nationality"]
)

# Create drifted data (for testing)
drifted_data = create_synthetic_drift_data(data, drift_magnitude=0.3)

# Detect drift
results = detector.detect_data_drift(drifted_data)
alert = detector.generate_drift_alert(results)

print(f"Drift detected: {results['dataset_drift_detected']}")
print(f"Alert level: {alert['alert_level']}")
```

### 3. MLflow Operations

```python
from src.mlflow_config import MLflowTracker

# Initialize tracker
tracker = MLflowTracker()

# Log a model run
run_id = tracker.log_model_run(
    model_name="my_model",
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    hyperparams={"param1": "value1"}
)

# Register best model
model_version = tracker.register_best_model(
    model_name="my_model",
    run_id=run_id,
    stage="Production"
)

# Compare runs
comparison_df = tracker.compare_runs()
```

## 📊 MLflow UI Features

Access the MLflow UI at `http://localhost:5000` to view:

1. **Experiments**: All pipeline runs with metrics comparison
2. **Models**: Registered models with version history
3. **Artifacts**: Stored model files and metadata
4. **Metrics**: Interactive metric visualization and comparison
5. **Parameters**: Hyperparameter tracking across runs

## 🚨 Drift Monitoring

### Automated Monitoring
- **Statistical Tests**: KS test for numerical, Chi-square for categorical
- **Thresholds**: Configurable significance levels (default: 0.05)
- **Alert Triggers**: Based on drift share percentage

### Alert Levels
- **LOW**: <15% of features drifting (no action needed)
- **MEDIUM**: 15-50% of features drifting (monitor closely)
- **HIGH**: >50% of features drifting (immediate action required)

### Recommendations
- Review data collection process
- Consider model retraining
- Investigate root causes
- Update feature engineering pipeline

## 🐳 Deployment

### Docker Support
The project includes Docker configurations for:
- **Airflow**: Complete workflow orchestration
- **MLflow**: Tracking server deployment
- **Application**: Containerized ML pipeline

### Airflow DAG
- **Schedule**: Weekly execution
- **Tasks**: Data validation, drift detection, conditional retraining
- **Monitoring**: Integrated alerting and cleanup

## 📋 Generated Reports

1. **MLflow Run Comparison** (`reports/mlflow_run_comparison.csv`)
   - All runs with metrics and parameters
   - Sorted by performance for easy comparison

2. **Drift Report** (`reports/drift_report.json`)
   - Detailed drift analysis per feature
   - Data quality assessment results
   - Alert recommendations

3. **Traditional Reports**
   - Model metrics and confusion matrices
   - Visualization exports (ROC curves, distributions)

## 🔍 Monitoring Dashboard

The MLflow UI provides:
- Real-time experiment tracking
- Model performance comparison
- Artifact management
- Model registry operations

## 🛠️ Troubleshooting

### Common Issues

1. **MLflow Not Starting**
   ```bash
   # Check if port 5000 is available
   netstat -an | grep 5000
   
   # Use different port if needed
   mlflow ui --port 5001
   ```

2. **Drift Detection Errors**
   - Ensure scipy is installed: `pip install scipy`
   - Check feature names match between reference and current data

3. **Permission Issues**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   ```

## 🔄 Next Steps

### Potential Enhancements
1. **Real-time Monitoring**: Stream processing for live drift detection
2. **Advanced Drift Detection**: More sophisticated algorithms (population stability index, etc.)
3. **A/B Testing Integration**: Champion/challenger model deployment
4. **Automated Retraining**: Trigger-based model retraining workflows
5. **Dashboard Integration**: Custom monitoring dashboards
6. **Multi-environment Support**: Dev/staging/production deployment strategies

### Production Considerations
1. **Scalability**: Database backend for MLflow (PostgreSQL/MySQL)
2. **Security**: Authentication and authorization for MLflow UI
3. **Storage**: Remote artifact storage (S3, Azure Blob, GCS)
4. **Monitoring**: Production monitoring and alerting systems

## 📚 Documentation

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **Evidently (alternative)**: https://evidently.ai/
- **Airflow Documentation**: https://airflow.apache.org/docs/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

---

**Note**: This integration maintains backward compatibility with the original pipeline while adding comprehensive experiment tracking and drift monitoring capabilities.
