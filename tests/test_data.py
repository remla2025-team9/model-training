import pytest
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from src.features import generate_features, split_and_save_data
from src import config

# ---- Fixtures ----

@pytest.fixture(scope="module")
def mock_dataset():
    data = {
        "Review": [
            "I love this product", "Terrible experience", "Okay, not great", 
            "Fantastic! Will buy again", "Worst purchase ever", "Loved it", 
            "Horrible customer service", "Very satisfied", "Awful!", "Highly recommend"
        ],
        "Liked": [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def processed_features_and_labels(mock_dataset):
    X, y = generate_features(mock_dataset, text_col="Review", label_col="Liked")
    return X, y

# ---- Tests for generate_features ----

def test_generate_features_returns_valid_output(processed_features_and_labels):
    X, y = processed_features_and_labels
    assert issparse(X), "Expected sparse matrix for features"
    assert X.shape[0] == len(y), "Mismatch between number of samples and labels"
    assert set(np.unique(y)).issubset({0, 1}), "Labels must be binary (0 or 1)"
    assert X.shape[1] > 0, "Feature matrix should have non-zero columns"

def test_features_have_no_nan_or_inf(processed_features_and_labels):
    X, _ = processed_features_and_labels
    dense = X.toarray()
    assert not np.isnan(dense).any(), "Features contain NaN"
    assert not np.isinf(dense).any(), "Features contain Inf"
    assert (dense >= 0).all(), "Features should not contain negative values"

def test_label_distribution(processed_features_and_labels):
    _, y = processed_features_and_labels
    unique = np.unique(y)
    assert len(unique) > 1, "Should contain at least two classes"

def test_basic_correlation(processed_features_and_labels):
    X, y = processed_features_and_labels
    dense = X.toarray()
    pos_mean = dense[y == 1].mean(axis=0)
    neg_mean = dense[y == 0].mean(axis=0)
    difference = np.abs(pos_mean - neg_mean)
    assert np.any(difference > 0.01), "No meaningful feature difference between classes"

# ---- Tests for input validation ----

def test_invalid_label_column_raises_error(mock_dataset):
    with pytest.raises(KeyError):
        generate_features(mock_dataset, text_col="Review", label_col="InvalidLabel")

def test_invalid_text_column_raises_error(mock_dataset):
    with pytest.raises(KeyError):
        generate_features(mock_dataset, text_col="InvalidText", label_col="Liked")

# ---- Test for data splitting and saving (side-effect test) ----

def test_split_and_save_data_creates_files(tmp_path, processed_features_and_labels):
    X, y = processed_features_and_labels
    
    # Temporarily override config paths
    config.PROCESSED_DATA_DIR = tmp_path
    config.TRAIN_FEATURES_FILE = tmp_path / "train_features.npz"
    config.TEST_FEATURES_FILE = tmp_path / "test_features.npz"
    config.TRAIN_LABELS_FILE = tmp_path / "train_labels.csv"
    config.TEST_LABELS_FILE = tmp_path / "test_labels.csv"

    split_and_save_data(X, y, test_size=0.3, random_state=123)

    # Check file existence
    assert config.TRAIN_FEATURES_FILE.exists(), "Train features file not created"
    assert config.TEST_FEATURES_FILE.exists(), "Test features file not created"
    assert config.TRAIN_LABELS_FILE.exists(), "Train labels file not created"
    assert config.TEST_LABELS_FILE.exists(), "Test labels file not created"
