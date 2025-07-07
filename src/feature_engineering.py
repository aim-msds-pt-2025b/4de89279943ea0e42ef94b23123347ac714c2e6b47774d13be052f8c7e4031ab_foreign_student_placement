import pandas as pd
from sklearn.cluster import KMeans


def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    1) interactions, ratios, polynomials
    2) bucketize & count-encode & combine cats
    3) clustering
    4) one-hot & align
    """
    for df in (X_train, X_test):
        df["study_duration"] = df["graduation_year"] - df["year_of_enrollment"]
        df["gpa_test_interaction"] = df["gpa_or_score"] * df["test_score"]
        df["study_duration_sq"] = df["study_duration"] ** 2
        df["gpa_test_ratio"] = df["gpa_or_score"] / (df["test_score"] + 1e-3)

        df["gpa_bucket"] = pd.qcut(df["gpa_or_score"], 3, labels=False)
        df["test_bucket"] = pd.qcut(df["test_score"], 3, labels=False)

        freq = df["enrollment_reason"].value_counts()
        df["enroll_reason_count"] = (
            df["enrollment_reason"].map(freq).fillna(0).astype(int)
        )

        df["scholarship_visa_combo"] = (
            df["scholarship_received"].astype(str) + "_" + df["visa_status"].astype(str)
        )

    cols = ["gpa_or_score", "test_score", "study_duration"]
    kmeans = KMeans(n_clusters=5, random_state=42)
    X_train["num_cluster"] = kmeans.fit_predict(X_train[cols])
    X_test["num_cluster"] = kmeans.predict(X_test[cols])

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test
