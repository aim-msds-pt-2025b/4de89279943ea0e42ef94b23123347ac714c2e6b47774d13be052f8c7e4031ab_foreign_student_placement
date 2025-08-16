#!/bin/bash

# MLflow UI Launcher Script
# This script starts the MLflow tracking UI

echo "🚀 Starting MLflow Tracking UI..."

# Set MLflow backend store URI (local file-based)
export MLFLOW_BACKEND_STORE_URI="file://./mlruns"

# Set MLflow default artifact root (local file-based)
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlruns"

# Check if mlruns directory exists, create if not
if [ ! -d "mlruns" ]; then
    echo "📁 Creating mlruns directory..."
    mkdir -p mlruns
fi

echo "📊 MLflow UI will be available at: http://localhost:5000"
echo "🔍 Backend store: $MLFLOW_BACKEND_STORE_URI"
echo "📁 Artifact root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo ""
echo "💡 Tips:"
echo "   • Use Ctrl+C to stop the server"
echo "   • Run your ML pipeline to see experiments populate"
echo "   • Access the Model Registry to manage model versions"
echo ""

# Start MLflow UI
mlflow ui --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" --host 0.0.0.0 --port 5000
