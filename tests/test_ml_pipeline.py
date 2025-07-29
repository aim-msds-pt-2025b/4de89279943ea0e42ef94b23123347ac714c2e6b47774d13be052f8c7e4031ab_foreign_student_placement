# tests/test_ml_pipeline.py

import sys
import os
import pandas as pd
import numpy as np

# 1) make sure pytest can see your src/ folder
HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_base_models, tune_models, build_ensemble
from evaluation import evaluate_models, select_and_save_best


DATA_CSV = os.path.abspath(
    os.path.join(HERE, "..", "data", "global_student_migration.csv")
)


def test_preprocess_data_loads_and_splits():
    X_train, X_test, y_train, y_test = preprocess_data(DATA_CSV)
    # sanity
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert not X_train.empty
    assert set(y_train.unique()) <= {0, 1}
    # ensure split
    assert len(X_train) + len(X_test) == len(pd.read_csv(DATA_CSV))


def test_engineer_features_preserves_shape():
    X_train, X_test, _, _ = preprocess_data(DATA_CSV)
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)
    # same number of rows, at least same number of columns
    assert X_train_fe.shape[0] == X_train.shape[0]
    assert X_test_fe.shape[0] == X_test.shape[0]
    assert X_train_fe.shape[1] >= X_train.shape[1]


def test_train_and_tune_and_ensemble(tmp_path):
    # 1) preprocess & engineer
    X_train, X_test, y_train, y_test = preprocess_data(DATA_CSV)
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)

    # 2) train base models
    model_dir = tmp_path / "models"
    train_base_models(X_train_fe, y_train, models_dir=str(model_dir))
    # expect at least one .joblib file
    files = list(model_dir.glob("*.pkl"))
    assert files, "No model files saved"

    # 3) hyperparameter tuning
    best = tune_models(X_train_fe, y_train)
    assert isinstance(best, dict)
    assert all(hasattr(m, "fit") for m in best.values())

    # 4) build ensemble
    ensemble = build_ensemble(best, X_train_fe, y_train, models_dir=str(model_dir))
    assert hasattr(ensemble, "predict"), "Ensemble has no predict"


def test_evaluate_and_select(tmp_path):
    # build a dummy classifier that predicts majority class
    from sklearn.dummy import DummyClassifier

    X_train, X_test, y_train, y_test = preprocess_data(DATA_CSV)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    models = {"dummy": dummy}

    # evaluate_models should return a DataFrame
    df = evaluate_models(models, X_test, y_test)
    assert "accuracy" in df.columns and "f1_score" in df.columns
    assert df.index.tolist() == ["dummy"]

    # select_and_save_best writes out metrics.txt
    out = tmp_path / "reports" / "metrics.txt"
    best_name, cm = select_and_save_best(
        df, models, X_test, y_test, metrics_txt_path=str(out)
    )
    assert out.exists()
    contents = out.read_text()
    assert f"Best model: {best_name}" in contents
    # confusion_matrix is a numpy array
    assert isinstance(cm, np.ndarray)
