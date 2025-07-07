import os
import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, model_path: str = "../models/model.pkl"):
    """
    Train a RandomForest and save to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return clf
