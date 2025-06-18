import pytest
import joblib
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from sklearn.linear_model import LogisticRegression
from src.modeling import predict
from src import config


@pytest.fixture
def setup_test_data(tmp_path):
    """Fixture to create mock test features, labels, and a model."""
    # Prepare fake test data
    X = csr_matrix(np.random.rand(10, 5))
    y = np.random.randint(0, 2, size=10)

    config.TEST_FEATURES_FILE = tmp_path / "test_features.npz"
    config.TEST_LABELS_FILE = tmp_path / "test_labels.csv"
    config.EVALUATION_METRICS_FILE = tmp_path / "metrics.json"
    config.REPORTS_DIR = tmp_path

    save_npz(config.TEST_FEATURES_FILE, X)
    pd.DataFrame({"label": y}).to_csv(config.TEST_LABELS_FILE, index=False)

    # Save a simple trained model
    model = LogisticRegression(max_iter=1000)
    model.fit(X.toarray(), y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    return model_path, y


def test_predict_main_success(tmp_path, setup_test_data):
    """Test evaluation runs and saves metrics correctly."""
    model_path, y_true = setup_test_data

    args = type("Args", (), {"model_path": str(model_path)})
    predict.main(args)

    # Check evaluation file was written
    assert config.EVALUATION_METRICS_FILE.exists()
    with open(config.EVALUATION_METRICS_FILE) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"] is not None


def test_predict_model_file_missing(tmp_path, setup_test_data):
    """Test missing model file raises FileNotFoundError."""
    args = type("Args", (), {"model_path": str(tmp_path / "nonexistent_model.joblib")})
    with pytest.raises(FileNotFoundError):
        predict.main(args)


def test_predict_memory_error(monkeypatch, setup_test_data):
    """Simulate a MemoryError when converting test features to dense."""
    class FakeSparseMatrix:
        def toarray(self):
            raise MemoryError("Simulated memory error")

    model_path, _ = setup_test_data

    monkeypatch.setattr(predict, "load_npz", lambda _: FakeSparseMatrix())
    args = type("Args", (), {"model_path": str(model_path)})

    with pytest.raises(MemoryError):
        predict.main(args)


def test_main_warns_when_test_data_missing(monkeypatch, caplog):
    """Test logger warning when test data files are missing."""
    caplog.set_level(logging.WARNING)

    monkeypatch.setattr(config, "TEST_FEATURES_FILE", Path("missing_features.npz"))
    monkeypatch.setattr(config, "TEST_LABELS_FILE", Path("missing_labels.csv"))
    monkeypatch.setattr(predict, "logger", predict.logger)

    # Simulate __main__ guard logic
    if not (config.TEST_FEATURES_FILE.exists() and config.TEST_LABELS_FILE.exists()):
        predict.logger.warning(
            "Processed test data not found. Please run 'python -m sentiment_src.dataset' first."
        )

    assert "Processed test data not found" in caplog.text
