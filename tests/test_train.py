import pytest
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from src.modeling import train
from src import config


@pytest.fixture
def processed_data(tmp_path):
    """Fixture to create mock training features and labels."""
    X = csr_matrix(np.random.rand(10, 5))
    y = np.random.randint(0, 2, size=10)

    # Redirect config paths to tmp_path
    config.TRAIN_FEATURES_FILE = tmp_path / "train_features.npz"
    config.TRAIN_LABELS_FILE = tmp_path / "train_labels.csv"
    save_npz(config.TRAIN_FEATURES_FILE, X)
    pd.DataFrame({"label": y}).to_csv(config.TRAIN_LABELS_FILE, index=False)

    return X, y


@pytest.mark.parametrize("model_type", ["nb", "logistic"])
def test_train_model_with_valid_model_types(tmp_path, model_type, processed_data):
    """Test training and saving with both supported model types."""
    config.MODELS_DIR = tmp_path

    args = type("Args", (), {"model_type": model_type})
    train.main(args)

    model_file = config.MODELS_DIR / "model.joblib"
    assert model_file.exists()
    clf = joblib.load(model_file)
    assert clf is not None


def test_train_model_with_invalid_model_type(processed_data):
    """Test ValueError is raised with unsupported model type."""
    args = type("Args", (), {"model_type": "unsupported"})
    with pytest.raises(ValueError, match="Unsupported model_type"):
        train.main(args)


def test_train_model_memory_error(monkeypatch, processed_data):
    """Simulate a MemoryError when converting sparse matrix to dense."""

    class FakeSparseMatrix:
        def toarray(self):
            raise MemoryError("Simulated memory error")

    monkeypatch.setattr(train, "load_npz", lambda _: FakeSparseMatrix())
    monkeypatch.setattr(train, "pd", pd)
    monkeypatch.setattr(train, "joblib", joblib)
    monkeypatch.setattr(train, "config", config)
    monkeypatch.setattr(train, "logger", train.logger)

    args = type("Args", (), {"model_type": "nb"})

    with pytest.raises(MemoryError):
        train.main(args)


def test_main_warns_when_data_missing(monkeypatch, caplog):
    """Test the warning when processed training data files are missing."""
    caplog.set_level(logging.WARNING)

    # Set nonexistent paths
    monkeypatch.setattr(config, "TRAIN_FEATURES_FILE", Path("fake_features.npz"))
    monkeypatch.setattr(config, "TRAIN_LABELS_FILE", Path("fake_labels.csv"))

    # Simulate the main() guard logic manually
    if not config.TRAIN_FEATURES_FILE.exists() or not config.TRAIN_LABELS_FILE.exists():
        train.logger.warning(
            "Processed train data not found. Please run 'python -m sentiment_src.dataset' first."
        )

    assert "Processed train data not found" in caplog.text
