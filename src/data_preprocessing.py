import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from drift_detection import create_synthetic_drift_data
except Exception:  # optional
    create_synthetic_drift_data = None


def preprocess_data(
    path: str,
    test_size=0.2,
    random_state: int = 42,
    emit_drifted: bool = False,
    drift_magnitude: float = 0.2,
    reports_dir: str = "reports",
):
    """
    1) Load CSV (with keep_default_na=False so “None” stays a string)
    2) Drop identifier/leakage columns
    3) Map target to binary
    4) Drop rows with missing core columns
    5) Split into train/test
    6) Scale numeric features
    """
    # 1) Load, disabling default NA inference so “None” isn’t turned into NaN
    df = pd.read_csv(path, keep_default_na=False)

    # 2) Drop leakage/ID columns (ignore if missing)
    drop_cols = [
        "student_id",
        "destination_city",
        "university_name",
        "course_name",
        "placement_country",
        "placement_company",
        "starting_salary_usd",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 3) Map target to 0/1
    df["placement_status"] = df["placement_status"].map({"Placed": 1, "Not Placed": 0})

    # 4) Drop any rows missing our “core” fields
    core = [
        "placement_status",
        "gpa_or_score",
        "test_score",
        "field_of_study",
        "origin_country",
        "destination_country",
        "scholarship_received",
        "enrollment_reason",
        "language_proficiency_test",
        "visa_status",
        "post_graduation_visa",
        "graduation_year",
        "year_of_enrollment",
    ]
    df = df.dropna(subset=core)

    # 5) Split into X/y and then train vs. test
    X = df.drop(columns="placement_status")
    y = df["placement_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6) Scale numeric columns only; keep categoricals as strings for later one-hot
    num_cols = ["gpa_or_score", "test_score", "year_of_enrollment", "graduation_year"]
    scaler = StandardScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 7) Generate drifted data as per homework requirements
    if emit_drifted:
        # Generate drifted training data
        X_train_drifted = X_train.copy()
        y_train_drifted = y_train.copy()

        # Apply drift to numerical features: multiply by 1.2 or add Gaussian noise
        for col in num_cols:
            # Stronger deterministic drift: scale and add small noise
            feature_std = X_train[col].std()
            noise = np.random.normal(0, 0.05 * feature_std, len(X_train_drifted))
            X_train_drifted[col] = X_train_drifted[col] * 1.3 + noise

        # Apply drift to categorical features: randomly flip 10-15% of values
        cat_cols = [col for col in X_train.columns if col not in num_cols]
        for col in cat_cols:
            unique_values = X_train[col].unique()
            if len(unique_values) > 1:  # Only if there are multiple categories
                flip_mask = (
                    np.random.random(len(X_train_drifted)) < 0.3
                )  # 30% flip rate
                for idx in X_train_drifted[flip_mask].index:
                    current_val = X_train_drifted.loc[idx, col]
                    other_vals = [v for v in unique_values if v != current_val]
                    if other_vals:
                        X_train_drifted.loc[idx, col] = np.random.choice(other_vals)

        # Generate drifted test data (same logic)
        X_test_drifted = X_test.copy()
        y_test_drifted = y_test.copy()

        # Apply drift to numerical features
        for col in num_cols:
            feature_std = X_train[col].std()  # Use training std for consistency
            noise = np.random.normal(0, 0.05 * feature_std, len(X_test_drifted))
            X_test_drifted[col] = X_test_drifted[col] * 1.3 + noise

        # Apply drift to categorical features
        for col in cat_cols:
            unique_values = X_train[col].unique()
            if len(unique_values) > 1:
                flip_mask = np.random.random(len(X_test_drifted)) < 0.3
                for idx in X_test_drifted[flip_mask].index:
                    current_val = X_test_drifted.loc[idx, col]
                    other_vals = [v for v in unique_values if v != current_val]
                    if other_vals:
                        X_test_drifted.loc[idx, col] = np.random.choice(other_vals)

        # Save drifted datasets as required
        os.makedirs("data", exist_ok=True)
        drifted_train = pd.concat([X_train_drifted, y_train_drifted], axis=1)
        drifted_test = pd.concat([X_test_drifted, y_test_drifted], axis=1)

        drifted_train.to_csv("data/drifted_train.csv", index=False)
        drifted_test.to_csv("data/drifted_test.csv", index=False)

        # Return tuple as specified in homework
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_drifted,
            y_train_drifted,
            X_test_drifted,
            y_test_drifted,
        )

    return X_train, X_test, y_train, y_test
