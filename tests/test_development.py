import pytest
import numpy as np
from scipy.sparse import issparse
from src import dataset
from src import config  
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

REAL_RAW_DATA_PATH = config.RAW_DATA_DIR / "training_data.tsv"

@pytest.fixture(scope="module")
def real_raw_data_path():
    return str(REAL_RAW_DATA_PATH)

@pytest.fixture(scope="module")
def processed_data(real_raw_data_path):
    X, y = dataset.load_and_generate_features(real_raw_data_path, text_col='Review', label_col='Liked')
    return X, y

def get_latest_model_file(models_dir: Path):
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return model_files[0]

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def test_model_vs_baseline(processed_data):
    """Test that the trained model performs better than the dummy baseline."""
    X, y = processed_data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Baseline dummy classifier
    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy.fit(X_train, y_train)
    baseline_preds = dummy.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    # Load latest trained model
    models_dir = config.MODELS_DIR
    model_path = get_latest_model_file(models_dir)
    model = joblib.load(model_path)
    model_preds = model.predict(X_test)
    model_acc = accuracy_score(y_test, model_preds)

    assert model_acc > baseline_acc, (
        f"Model did not outperform baseline. "
        f"Model accuracy: {model_acc:.3f}, Baseline accuracy: {baseline_acc:.3f}"
    )
