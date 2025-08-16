"""Simple data drift detection using statistical methods."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
from scipy import stats


class SimpleDriftDetector:
    """Simple statistical data drift detection."""

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

        # Calculate reference statistics
        self.reference_stats = self._calculate_stats(reference_data)

    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for a dataset."""
        stats_dict = {}

        # Numerical features
        for feature in self.numerical_features:
            if feature in data.columns:
                stats_dict[feature] = {
                    "mean": data[feature].mean(),
                    "std": data[feature].std(),
                    "median": data[feature].median(),
                    "min": data[feature].min(),
                    "max": data[feature].max(),
                    "q25": data[feature].quantile(0.25),
                    "q75": data[feature].quantile(0.75),
                }

        # Categorical features
        for feature in self.categorical_features:
            if feature in data.columns:
                value_counts = data[feature].value_counts(normalize=True)
                stats_dict[feature] = {
                    "value_counts": value_counts.to_dict(),
                    "unique_count": data[feature].nunique(),
                    "top_value": (
                        value_counts.index[0] if len(value_counts) > 0 else None
                    ),
                    "top_freq": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                }

        return stats_dict

    def _ks_test_drift(
        self, ref_data: pd.Series, current_data: pd.Series
    ) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for numerical drift."""
        try:
            statistic, p_value = stats.ks_2samp(
                ref_data.dropna(), current_data.dropna()
            )
            return statistic, p_value
        except Exception:
            return 0.0, 1.0

    def _chi2_test_drift(
        self, ref_data: pd.Series, current_data: pd.Series
    ) -> Tuple[float, float]:
        """Perform Chi-square test for categorical drift."""
        try:
            # Get value counts for both datasets
            ref_counts = ref_data.value_counts()
            current_counts = current_data.value_counts()

            # Align categories
            all_categories = set(ref_counts.index) | set(current_counts.index)
            ref_aligned = pd.Series([ref_counts.get(cat, 0) for cat in all_categories])
            current_aligned = pd.Series(
                [current_counts.get(cat, 0) for cat in all_categories]
            )

            # Perform chi-square test
            statistic, p_value = stats.chisquare(current_aligned, f_exp=ref_aligned)
            return statistic, p_value
        except Exception:
            return 0.0, 1.0

    def detect_data_drift(
        self, current_data: pd.DataFrame, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """Detect data drift between reference and current data.

        Args:
            current_data: Current/new dataset
            significance_level: Statistical significance level

        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "significance_level": significance_level,
            "drifted_features": [],
            "feature_drift_scores": {},
            "dataset_drift_detected": False,
            "drift_share": 0.0,
            "number_of_drifted_columns": 0,
        }

        total_features = 0
        drifted_count = 0

        # Check numerical features
        for feature in self.numerical_features:
            if (
                feature in self.reference_data.columns
                and feature in current_data.columns
            ):
                total_features += 1

                ref_series = self.reference_data[feature]
                current_series = current_data[feature]

                statistic, p_value = self._ks_test_drift(ref_series, current_series)
                drift_detected = p_value < significance_level

                drift_results["feature_drift_scores"][feature] = {
                    "test": "KS_test",
                    "statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": drift_detected,
                    "drift_score": statistic,  # Use KS statistic as drift score
                }

                if drift_detected:
                    drifted_count += 1
                    drift_results["drifted_features"].append(
                        {
                            "feature": feature,
                            "drift_score": statistic,
                            "p_value": p_value,
                            "test": "KS_test",
                        }
                    )

        # Check categorical features
        for feature in self.categorical_features:
            if (
                feature in self.reference_data.columns
                and feature in current_data.columns
            ):
                total_features += 1

                ref_series = self.reference_data[feature]
                current_series = current_data[feature]

                statistic, p_value = self._chi2_test_drift(ref_series, current_series)
                drift_detected = p_value < significance_level

                drift_results["feature_drift_scores"][feature] = {
                    "test": "Chi2_test",
                    "statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": drift_detected,
                    "drift_score": min(
                        statistic / 100, 1.0
                    ),  # Normalize chi2 statistic
                }

                if drift_detected:
                    drifted_count += 1
                    drift_results["drifted_features"].append(
                        {
                            "feature": feature,
                            "drift_score": min(statistic / 100, 1.0),
                            "p_value": p_value,
                            "test": "Chi2_test",
                        }
                    )

        # Calculate overall drift metrics
        if total_features > 0:
            drift_results["drift_share"] = drifted_count / total_features
            drift_results["number_of_drifted_columns"] = drifted_count
            drift_results["dataset_drift_detected"] = (
                drift_results["drift_share"] > 0.1
            )  # 10% threshold

        return drift_results

    def detect_target_drift(
        self, current_data: pd.DataFrame, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """Detect target variable drift.

        Args:
            current_data: Current dataset with target
            significance_level: Statistical significance level

        Returns:
            Target drift results
        """
        if self.target_column not in current_data.columns:
            return {
                "error": f"Target column '{self.target_column}' not found in current data"
            }

        ref_target = self.reference_data[self.target_column]
        current_target = current_data[self.target_column]

        # Check if target is numerical or categorical
        if ref_target.dtype in ["int64", "float64"]:
            statistic, p_value = self._ks_test_drift(ref_target, current_target)
            test_type = "KS_test"
        else:
            statistic, p_value = self._chi2_test_drift(ref_target, current_target)
            test_type = "Chi2_test"

        return {
            "timestamp": datetime.now().isoformat(),
            "target_drift_detected": p_value < significance_level,
            "target_drift_score": statistic,
            "p_value": p_value,
            "test_type": test_type,
            "significance_level": significance_level,
        }

    def run_data_quality_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Run basic data quality tests on current data.

        Args:
            current_data: Current dataset

        Returns:
            Data quality test results
        """
        quality_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": [],
        }

        # Test 1: Check for missing values
        missing_counts = current_data.isnull().sum()
        missing_test_passed = missing_counts.sum() == 0

        quality_results["test_details"].append(
            {
                "test_name": "No missing values",
                "status": "SUCCESS" if missing_test_passed else "FAILED",
                "description": f"Total missing values: {missing_counts.sum()}",
            }
        )

        if missing_test_passed:
            quality_results["tests_passed"] += 1
        else:
            quality_results["tests_failed"] += 1

        # Test 2: Check for constant columns
        constant_cols = []
        for col in current_data.columns:
            if current_data[col].nunique() <= 1:
                constant_cols.append(col)

        constant_test_passed = len(constant_cols) == 0

        quality_results["test_details"].append(
            {
                "test_name": "No constant columns",
                "status": "SUCCESS" if constant_test_passed else "FAILED",
                "description": f"Constant columns: {constant_cols}",
            }
        )

        if constant_test_passed:
            quality_results["tests_passed"] += 1
        else:
            quality_results["tests_failed"] += 1

        # Test 3: Check for duplicated rows
        duplicate_count = current_data.duplicated().sum()
        duplicate_test_passed = duplicate_count == 0

        quality_results["test_details"].append(
            {
                "test_name": "No duplicated rows",
                "status": "SUCCESS" if duplicate_test_passed else "FAILED",
                "description": f"Duplicated rows: {duplicate_count}",
            }
        )

        if duplicate_test_passed:
            quality_results["tests_passed"] += 1
        else:
            quality_results["tests_failed"] += 1

        # Test 4: Check data shape consistency
        shape_test_passed = len(current_data.columns) == len(
            self.reference_data.columns
        )

        quality_results["test_details"].append(
            {
                "test_name": "Consistent data shape",
                "status": "SUCCESS" if shape_test_passed else "FAILED",
                "description": f"Current: {current_data.shape}, Reference: {self.reference_data.shape}",
            }
        )

        if shape_test_passed:
            quality_results["tests_passed"] += 1
        else:
            quality_results["tests_failed"] += 1

        return quality_results

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
            json.dump(drift_results, f, indent=2, default=str)

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


# For backward compatibility, alias the class
DriftDetector = SimpleDriftDetector
