import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(path: str, test_size: float = 0.2, random_state: int = 42):
    """
    1) Load CSV
    2) Drop identifier/leakage columns
    3) Map target to binary
    4) Drop rows with missing core columns
    5) Split into train/test
    6) Scale numeric features
    """
    df = pd.read_csv(path)

    # drop leakage columns
    drop_cols = [
        "student_id",
        "destination_city",
        "university_name",
        "course_name",
        "placement_country",
        "placement_company",
        "starting_salary_usd",
    ]
    df = df.drop(drop_cols, axis=1)

    # binary target
    df["placement_status"] = df["placement_status"].map({"Placed": 1, "Not Placed": 0})

    # drop missing in core
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

    X = df.drop("placement_status", axis=1)
    y = df["placement_status"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # scale numeric columns
    num_cols = ["gpa_or_score", "test_score", "year_of_enrollment", "graduation_year"]
    scaler = StandardScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test
