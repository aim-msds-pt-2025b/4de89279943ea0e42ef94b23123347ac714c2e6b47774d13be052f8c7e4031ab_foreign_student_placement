# tests/test_data_preprocessing.py

import os
import sys
import pandas as pd

# Make sure we import from src/
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from data_preprocessing import preprocess_data


def test_preprocess_data(tmp_path):
    # 1) Create a minimal CSV with 6 rows (3 Placed, 3 Not Placed)
    df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 4, 5, 6],
            "destination_city": ["X", "Y", "Z", "W", "X", "Y"],
            "university_name": ["U1", "U2", "U3", "U4", "U5", "U6"],
            "course_name": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "placement_country": ["P"] * 6,
            "placement_company": ["Comp"] * 6,
            "starting_salary_usd": [1000, 2000, 3000, 4000, 5000, 6000],
            "placement_status": ["Placed", "Not Placed"] * 3,
            "gpa_or_score": [3.5, 2.8, 3.9, 2.5, 3.7, 2.6],
            "test_score": [80, 60, 90, 50, 85, 55],
            "field_of_study": ["Eng", "Arts", "Eng", "Arts", "Eng", "Arts"],
            "origin_country": ["A", "B", "A", "B", "A", "B"],
            "destination_country": ["C", "D", "C", "D", "C", "D"],
            "scholarship_received": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "enrollment_reason": ["R1", "R2", "R1", "R2", "R1", "R2"],
            "language_proficiency_test": [
                "TOEFL",
                "IELTS",
                "TOEFL",
                "IELTS",
                "TOEFL",
                "IELTS",
            ],
            "visa_status": ["Open", "Closed", "Open", "Closed", "Open", "Closed"],
            "post_graduation_visa": ["None"] * 6,
            "graduation_year": [2024, 2025, 2024, 2025, 2024, 2025],
            "year_of_enrollment": [2020, 2021, 2020, 2021, 2020, 2021],
        }
    )

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    # 2) Run loader asking for exactly 3 test samples (3 train / 3 test)
    X_train, X_test, y_train, y_test = preprocess_data(
        str(csv_path),
        test_size=3,
        random_state=0,
    )

    # 3) Check sizes: 3 rows in train, 3 in test
    assert len(X_train) == 3
    assert len(X_test) == 3
    assert len(y_train) == 3
    assert len(y_test) == 3

    # 4) Target mapping is 0/1
    assert set(y_train.unique()) <= {0, 1}
    assert set(y_test.unique()) <= {0, 1}

    # 5) Leakage columns are dropped
    for col in [
        "student_id",
        "destination_city",
        "university_name",
        "course_name",
        "placement_country",
        "placement_company",
        "starting_salary_usd",
    ]:
        assert col not in X_train.columns
        assert col not in X_test.columns
