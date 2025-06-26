import pytest
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from src import config
from pathlib import Path

def get_latest_model_file(models_dir: Path):
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return model_files[0]

# Load the trained sentiment classification model
@pytest.fixture(scope="module")
def trained_model():
    model_path = get_latest_model_file(config.MODELS_DIR)
    model = joblib.load(str(model_path))
    return model

# Load test features, labels, and raw test dataframe
@pytest.fixture(scope="module")
def test_data():
    X_test = load_npz(config.TEST_FEATURES_FILE)
    y_test = pd.read_csv(config.TEST_LABELS_FILE)['label']
    df_test = pd.read_csv(config.TEST_LABELS_FILE)
    return X_test, y_test, df_test

# Helper function to compute accuracy
def evaluate_accuracy(model, X, y):
    if hasattr(X, "toarray"):
        X = X.toarray()
    y_pred = model.predict(X)
    return (y_pred == y).mean()

# Test function: evaluate model accuracy on full dataset and slices
def test_model_accuracy_overall_and_slices(trained_model, test_data):
    X_test, y_test, df_test = test_data

    # Overall test accuracy
    overall_acc = evaluate_accuracy(trained_model, X_test, y_test)
    print(f"Overall test accuracy: {overall_acc:.3f}")
    assert overall_acc >= 0.5, "Overall accuracy is below 50%"

    # Accuracy on negative reviews (label = 0)
    neg_indices = np.where((y_test == 0).to_numpy())[0]
    neg_acc = evaluate_accuracy(trained_model, X_test[neg_indices], y_test.iloc[neg_indices])
    print(f"Negative review accuracy: {neg_acc:.3f}")
    assert neg_acc >= 0.5, "Negative review accuracy is below 50%"

    # Accuracy on positive reviews (label = 1)
    pos_indices = np.where((y_test == 1).to_numpy())[0]
    pos_acc = evaluate_accuracy(trained_model, X_test[pos_indices], y_test.iloc[pos_indices])
    print(f"Positive review accuracy: {pos_acc:.3f}")
    assert pos_acc >= 0.5, "Positive review accuracy is below 50%"
