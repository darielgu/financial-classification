from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from src.evaluate import compute_metrics, print_report, print_classification_report, save_confusion_matrix
from src.model_data import get_data_for_model

MODEL_NAME = "logistic_regression"
MODEL_PATH = Path("models/logistic_regression.joblib")

def main():

    print("Training logistic regression model...")

    x_train, y_train, x_val, y_val, x_test, y_test, _ = get_data_for_model()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
                               
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, model_name = MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    save_confusion_matrix(y_test, y_pred, model_name=MODEL_NAME, output_dir=MODEL_PATH.parent)

    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()