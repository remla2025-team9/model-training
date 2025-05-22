import pytest
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from src import dataset

@pytest.fixture
def sample_raw_data(tmp_path):
    """Create a sample raw TSV file with reviews and labels for testing."""
    data = {
        'Review': ['Good food', 'Bad service', 'Okay experience', 'Excellent!'],
        'Liked': [1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "sample_raw.tsv"
    df.to_csv(file_path, sep='\t', index=False)
    return str(file_path)

def test_load_and_generate_features_basic(sample_raw_data):
    """Test that features and labels load correctly, features are sparse, and label values are valid."""
    X, y = dataset.load_and_generate_features(sample_raw_data, text_col='Review', label_col='Liked')
    assert issparse(X), "Features should be a sparse matrix"
    assert len(y) == X.shape[0], "Number of labels should match number of samples"
    assert set(np.unique(y)).issubset({0,1}), "Labels should be binary (0 or 1)"
    assert X.shape[0] > 0 and X.shape[1] > 0, "Features should not be empty"

def test_no_nan_or_inf_in_features(sample_raw_data):
    """Test that feature matrix contains no NaN or infinite values and no negatives."""
    X, _ = dataset.load_and_generate_features(sample_raw_data, text_col='Review', label_col='Liked')
    dense = X.toarray()
    assert not np.isnan(dense).any(), "Features contain NaN"
    assert not np.isinf(dense).any(), "Features contain Inf"
    assert (dense >= 0).all(), "Features contain negative values"

def test_label_distribution(sample_raw_data):
    """Test that label distribution contains more than one class."""
    _, y = dataset.load_and_generate_features(sample_raw_data, text_col='Review', label_col='Liked')
    unique_labels = np.unique(y)
    assert len(unique_labels) > 1, "Labels should contain at least two classes"

def test_feature_correlation_with_label(sample_raw_data):
    """Test that features show some difference between positive and negative classes."""
    X, y = dataset.load_and_generate_features(sample_raw_data, text_col='Review', label_col='Liked')
    dense = X.toarray()
    pos_mean = dense[y == 1].mean(axis=0)
    neg_mean = dense[y == 0].mean(axis=0)
    assert np.any(np.abs(pos_mean - neg_mean) > 0.01), "No meaningful difference in features between classes"

def test_preprocess_performance(sample_raw_data):
    """Test that feature generation runs within an acceptable time limit."""
    import time
    start = time.time()
    dataset.load_and_generate_features(sample_raw_data, text_col='Review', label_col='Liked')
    duration = time.time() - start
    assert duration < 5, "Feature generation took too long"

