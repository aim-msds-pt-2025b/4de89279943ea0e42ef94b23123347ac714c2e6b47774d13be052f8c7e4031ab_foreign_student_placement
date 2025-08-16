"""Data drift detection using Evidently."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime
import warnings

try:
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfColumnsWithMissingValues,
        TestNumberOfRowsWithMissingValues,
    )
    from evidently.tests import TestNumberOfConstantColumns, TestNumberOfDuplicatedRows
    from evidently.tests import TestColumnsType

    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:
        # Try newer API structure
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metrics import (
            DataDriftPreset,
            DataQualityPreset,
            TargetDriftPreset,
        )
        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestNumberOfColumnsWithMissingValues,
            TestNumberOfRowsWithMissingValues,
        )
        from evidently.tests import (
            TestNumberOfConstantColumns,
            TestNumberOfDuplicatedRows,
        )
        from evidently.tests import TestColumnsType

        EVIDENTLY_AVAILABLE = True
    except ImportError:
        EVIDENTLY_AVAILABLE = False
        warnings.warn("Evidently not available. Install with: pip install evidently")


class DriftDetector:
    """Data drift detection and monitoring."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_column: str = "placement_status",
        numerical_features: Optional[list] = None,
        categorical_features: Optional[list] = None,
    ):
        """Initialize drift detector.

        Args:
            reference_data: Reference dataset (training data)
            target_column: Name of target column
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []

        # Set up column mapping for Evidently
        self.column_mapping = (
            ColumnMapping(
                target=target_column,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )
            if EVIDENTLY_AVAILABLE
            else None
        )

    def detect_data_drift(
        self, current_data: pd.DataFrame, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Detect data drift between reference and current data.

        Args:
            current_data: Current/new dataset
            confidence_level: Statistical confidence level

        Returns:
            Dictionary with drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            return {"error": "Evidently not available"}

        try:
            # Create data drift report
            data_drift_report = Report(
                metrics=[
                    DataDriftPreset(confidence=confidence_level),
                    DataQualityPreset(),
                ]
            )

            data_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Extract results
            report_dict = data_drift_report.as_dict()

            # Parse key metrics
            drift_results = {
                "timestamp": datetime.now().isoformat(),
                "dataset_drift_detected": report_dict["metrics"][0]["result"][
                    "dataset_drift"
                ],
                "drift_share": report_dict["metrics"][0]["result"]["drift_share"],
                "number_of_drifted_columns": report_dict["metrics"][0]["result"][
                    "number_of_drifted_columns"
                ],
                "confidence_level": confidence_level,
                "drifted_features": [],
            }

            # Get details for drifted features
            if "drift_by_columns" in report_dict["metrics"][0]["result"]:
                for feature, drift_info in report_dict["metrics"][0]["result"][
                    "drift_by_columns"
                ].items():
                    if drift_info.get("drift_detected", False):
                        drift_results["drifted_features"].append(
                            {
                                "feature": feature,
                                "drift_score": drift_info.get("drift_score", 0),
                                "threshold": drift_info.get("threshold", 0),
                            }
                        )

            return drift_results

        except Exception as e:
            return {"error": f"Drift detection failed: {str(e)}"}

    def detect_target_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect target variable drift.

        Args:
            current_data: Current dataset with target

        Returns:
            Target drift results
        """
        if not EVIDENTLY_AVAILABLE:
            return {"error": "Evidently not available"}

        try:
            target_drift_report = Report(metrics=[TargetDriftPreset()])

            target_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            report_dict = target_drift_report.as_dict()

            return {
                "timestamp": datetime.now().isoformat(),
                "target_drift_detected": report_dict["metrics"][0]["result"].get(
                    "target_drift", False
                ),
                "target_drift_score": report_dict["metrics"][0]["result"].get(
                    "target_drift_score", 0
                ),
            }

        except Exception as e:
            return {"error": f"Target drift detection failed: {str(e)}"}

    def run_data_quality_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Run data quality tests on current data.

        Args:
            current_data: Current dataset

        Returns:
            Data quality test results
        """
        if not EVIDENTLY_AVAILABLE:
            return {"error": "Evidently not available"}

        try:
            # Define data quality tests
            data_quality_tests = TestSuite(
                tests=[
                    TestNumberOfColumnsWithMissingValues(),
                    TestNumberOfRowsWithMissingValues(),
                    TestNumberOfConstantColumns(),
                    TestNumberOfDuplicatedRows(),
                    TestColumnsType(),
                ]
            )

            data_quality_tests.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            test_results = data_quality_tests.as_dict()

            # Parse test results
            quality_results = {
                "timestamp": datetime.now().isoformat(),
                "tests_passed": 0,
                "tests_failed": 0,
                "test_details": [],
            }

            for test in test_results["tests"]:
                test_name = test["name"]
                test_status = test["status"]

                if test_status == "SUCCESS":
                    quality_results["tests_passed"] += 1
                else:
                    quality_results["tests_failed"] += 1

                quality_results["test_details"].append(
                    {
                        "test_name": test_name,
                        "status": test_status,
                        "description": test.get("description", ""),
                    }
                )

            return quality_results

        except Exception as e:
            return {"error": f"Data quality tests failed: {str(e)}"}

    def save_drift_report(
        self,
        drift_results: Dict[str, Any],
        output_path: str = "reports/drift_report.json",
    ):
        """Save drift detection results to file.

        Args:
            drift_results: Drift detection results
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(drift_results, f, indent=2)

        print(f"Drift report saved to: {output_path}")

    def generate_drift_alert(
        self, drift_results: Dict[str, Any], alert_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Generate alerts based on drift detection results.

        Args:
            drift_results: Drift detection results
            alert_threshold: Threshold for triggering alerts

        Returns:
            Alert information
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_triggered": False,
            "alert_level": "LOW",
            "message": "No significant drift detected",
            "recommendations": [],
        }

        if drift_results.get("dataset_drift_detected", False):
            drift_share = drift_results.get("drift_share", 0)

            if drift_share >= alert_threshold:
                alert["alert_triggered"] = True
                alert["alert_level"] = "HIGH" if drift_share >= 0.5 else "MEDIUM"
                alert["message"] = (
                    f"Significant data drift detected! {drift_share:.2%} of features are drifting."
                )
                alert["recommendations"] = [
                    "Review data collection process",
                    "Consider retraining the model",
                    "Investigate root causes of drift",
                    "Update feature engineering pipeline if needed",
                ]
            elif drift_share >= alert_threshold * 0.5:
                alert["alert_triggered"] = True
                alert["alert_level"] = "MEDIUM"
                alert["message"] = (
                    f"Moderate data drift detected. {drift_share:.2%} of features are drifting."
                )
                alert["recommendations"] = [
                    "Monitor closely",
                    "Consider model performance evaluation",
                    "Prepare for potential retraining",
                ]

        return alert


def create_synthetic_drift_data(
    original_data: pd.DataFrame,
    drift_magnitude: float = 0.2,
    drift_features: Optional[list] = None,
) -> pd.DataFrame:
    """Create synthetic drifted data for testing.

    Args:
        original_data: Original dataset
        drift_magnitude: Magnitude of drift to introduce
        drift_features: Features to introduce drift in

    Returns:
        Dataset with synthetic drift
    """
    drifted_data = original_data.copy()

    if drift_features is None:
        # Select numeric columns for drift
        numeric_cols = drifted_data.select_dtypes(include=[np.number]).columns
        drift_features = numeric_cols.tolist()

    for feature in drift_features:
        if feature in drifted_data.columns and drifted_data[feature].dtype in [
            "int64",
            "float64",
        ]:
            # Add gaussian noise proportional to the feature's standard deviation
            noise = np.random.normal(
                0, drift_magnitude * drifted_data[feature].std(), len(drifted_data)
            )
            drifted_data[feature] = drifted_data[feature] + noise

    return drifted_data
