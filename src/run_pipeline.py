from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model
from evaluation import evaluate_model


def main():
    # 1) load & preprocess
    X_train, X_test, y_train, y_test = preprocess_data(
        "../data/global_student_migration.csv"
    )
    # 2) feature engineering
    X_train_fe, X_test_fe = engineer_features(X_train, X_test)
    # 3) train & save
    model = train_model(X_train_fe, y_train, model_path="../models/model.pkl")
    print("Model saved.")
    # 4) evaluate & save metrics
    metrics = evaluate_model(
        model, X_test_fe, y_test, metrics_path="../reports/metrics.txt"
    )
    print("Metrics written:", metrics)


if __name__ == "__main__":
    main()
