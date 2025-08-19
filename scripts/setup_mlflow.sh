#!/bin/bash

# Setup script for MLflow and Drift Detection Integration
# This script installs dependencies and sets up the environment

set -e  # Exit on any error

echo "🚀 Setting up MLflow and Drift Detection Environment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Install/upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install project dependencies
echo "📦 Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install core dependencies if requirements.txt doesn't exist
    pip install pandas scikit-learn joblib seaborn matplotlib mlflow evidently
fi

# Install project in development mode
echo "📦 Installing project in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p reports/figures
mkdir -p models
mkdir -p mlruns
mkdir -p logs

# Set up MLflow environment
echo "⚙️ Setting up MLflow environment..."
export MLFLOW_TRACKING_URI="file://./mlruns"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlruns"

# Initialize MLflow experiment
echo "🧪 Initializing MLflow experiment..."
python -c "
import mlflow
mlflow.set_tracking_uri('file://./mlruns')
try:
    experiment = mlflow.get_experiment_by_name('student_placement_prediction')
    if experiment is None:
        experiment_id = mlflow.create_experiment('student_placement_prediction')
        print(f'✅ Created MLflow experiment: student_placement_prediction (ID: {experiment_id})')
    else:
        print(f'✅ MLflow experiment already exists: {experiment.name} (ID: {experiment.experiment_id})')
except Exception as e:
    print(f'⚠️ Warning: Could not initialize MLflow experiment: {e}')
"

# Run a quick test
echo "🧪 Running quick test..."
python -c "
import pandas as pd
import sklearn
import mlflow
import evidently
print('✅ All core dependencies imported successfully')
print(f'   pandas: {pd.__version__}')
print(f'   scikit-learn: {sklearn.__version__}')
print(f'   mlflow: {mlflow.__version__}')
print(f'   evidently: {evidently.__version__}')
"

# Test data availability
if [ -f "data/global_student_migration.csv" ]; then
    echo "✅ Training data found"
    python -c "
import pandas as pd
df = pd.read_csv('data/global_student_migration.csv')
print(f'   Data shape: {df.shape}')
print(f'   Columns: {list(df.columns)}')
"
else
    echo "⚠️ Training data not found at data/global_student_migration.csv"
    echo "   Please ensure your data file is in the correct location"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "   1. Run the ML pipeline: python src/run_pipeline_mlflow.py"
echo "   2. Start MLflow UI: ./scripts/start_mlflow_ui.sh (or .bat on Windows)"
echo "   3. View results at: http://localhost:5000"
echo ""
echo "📊 Available Scripts:"
echo "   • Run pipeline: python src/run_pipeline_mlflow.py"
echo "   • Original pipeline: python src/run_pipeline.py"
echo "   • Start MLflow UI: ./scripts/start_mlflow_ui.sh"
echo "   • Run tests: pytest tests/"
echo ""
echo "📁 Key Directories:"
echo "   • MLflow tracking: ./mlruns"
echo "   • Model artifacts: ./models"
echo "   • Reports: ./reports"
echo "   • Figures: ./reports/figures"
