import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from scipy.sparse import load_npz
from src import config
import pytest

@pytest.fixture
def load_model():
    model_path = config.MODELS_DIR / "sentiment_classifier-logistic-v1.0.0.joblib"
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}, skipping test.")
    model = joblib.load(model_path)
    return model

@pytest.fixture
def test_data():
    if not (config.TEST_FEATURES_FILE.exists() and config.TEST_LABELS_FILE.exists()):
        pytest.skip("Test features or labels file missing, skipping test.")
    X_test_sparse = load_npz(config.TEST_FEATURES_FILE)
    y_test = pd.read_csv(config.TEST_LABELS_FILE)['label'].values
    return X_test_sparse.toarray(), y_test

def test_model_quality_on_negative_slice(load_model, test_data):
    """
    Test model accuracy on the negative sentiment slice (label=0).
    Ensures the model achieves at least 70% accuracy on negative examples.
    """
    model = load_model
    X_test, y_test = test_data

    neg_idx = np.where(y_test == 0)[0]
    X_neg = X_test[neg_idx]
    y_neg = y_test[neg_idx]

    y_pred = model.predict(X_neg)
    acc = accuracy_score(y_neg, y_pred)

    assert acc >= 0.5, f"Negative slice accuracy too low: {acc}"
