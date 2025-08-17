import os
import json
import pandas as pd
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

    if emit_drifted and create_synthetic_drift_data is not None:
        # Construct current data by adding synthetic drift to test split (for demo)
        current = pd.concat([X_test.copy(), y_test.rename("placement_status")], axis=1)
        drifted = create_synthetic_drift_data(
            current,
            drift_magnitude=drift_magnitude,
            drift_features=[
                "gpa_or_score",
                "test_score",
                "year_of_enrollment",
                "graduation_year",
            ],
        )
        os.makedirs(reports_dir, exist_ok=True)
        drift_meta = {
            "generated": True,
            "drift_magnitude": drift_magnitude,
            "rows": len(drifted),
        }
        with open(os.path.join(reports_dir, "drift_meta.json"), "w") as f:
            json.dump(drift_meta, f, indent=2)
        # return both original and drifted current datasets for downstream use
        X_drift = drifted.drop(columns=["placement_status"])  # type: ignore
        y_drift = drifted["placement_status"]  # type: ignore
        return X_train, X_test, y_train, y_test, X_drift, y_drift

    return X_train, X_test, y_train, y_test
