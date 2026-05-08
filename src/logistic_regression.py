from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from src.evaluate import compute_metrics, print_report, print_report, print_classification_report, save_confusion_matrix
from src.model_data import get_data_for_model

""" 
RUN python3.14 -m src.logistic_regression
"""

# Constants to identify model and storage
MODEL_NAME = "logistic_regression"
MODEL_PATH = Path("models/logistic_regression.joblib")


def main():

    """
    Performs model training pipeline
    1. Loads preprocessed data
    2. Trains a logistic regression model
    3. Evaluates the model on test data
    4. Saves the trained model
    """

    print("Training logistic regression model...")

    # Load preprocessed data
    x_train, y_train, x_val, y_val, x_test, y_test, _ = get_data_for_model()

    # Initialize model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    # Train model with training set                
    model.fit(x_train, y_train)

    # Generate predictiond for evaluation
    y_pred = model.predict(x_test)

    # Evaluate the model
    metrics = compute_metrics(y_test, y_pred, model_name = MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    # Create 'models' directory if it doesn't exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)

    # Save the confusion matrix
    save_confusion_matrix(y_test, y_pred, model_name=MODEL_NAME, output_dir=MODEL_PATH.parent)

    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
