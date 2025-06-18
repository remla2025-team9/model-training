import pytest
import numpy as np
from scipy.sparse import issparse
from src import dataset
from src import config  

REAL_RAW_DATA_PATH = config.RAW_DATA_DIR / "training_data.tsv"

@pytest.fixture(scope="module")
def real_raw_data_path():
    return str(REAL_RAW_DATA_PATH)

@pytest.fixture(scope="module")
def processed_data(real_raw_data_path):
    X, y = dataset.load_and_generate_features(real_raw_data_path, text_col='Review', label_col='Liked')
    return X, y

def test_load_and_generate_features_basic(processed_data):
    X, y = processed_data
    assert issparse(X), "Features should be a sparse matrix"
    assert len(y) == X.shape[0], "Number of labels should match number of samples"
    assert set(np.unique(y)).issubset({0,1}), "Labels should be binary (0 or 1)"
    assert X.shape[0] > 0 and X.shape[1] > 0, "Features should not be empty"

def test_no_nan_or_inf_in_features(processed_data):
    X, _ = processed_data
    dense = X.toarray()
    assert not np.isnan(dense).any(), "Features contain NaN"
    assert not np.isinf(dense).any(), "Features contain Inf"
    assert (dense >= 0).all(), "Features contain negative values"

def test_label_distribution(processed_data):
    _, y = processed_data
    unique_labels = np.unique(y)
    assert len(unique_labels) > 1, "Labels should contain at least two classes"

def test_feature_correlation_with_label(processed_data):
    X, y = processed_data
    dense = X.toarray()
    pos_mean = dense[y == 1].mean(axis=0)
    neg_mean = dense[y == 0].mean(axis=0)
    assert np.any(np.abs(pos_mean - neg_mean) > 0.01), "No meaningful difference in features between classes"

def test_preprocess_performance(real_raw_data_path):
    import time
    start = time.time()
    dataset.load_and_generate_features(real_raw_data_path, text_col='Review', label_col='Liked')
    duration = time.time() - start
    assert duration < 5, "Feature generation took too long"

def test_duplicate_review_liked_pairs(real_raw_data_path):
    import pandas as pd
    df = pd.read_csv(real_raw_data_path, sep='\t')
    df.columns = df.columns.str.strip()
    dup_count = df.duplicated(subset=['Review', 'Liked']).sum()
    assert dup_count <= 10, f"Too many duplicated review-label pairs: {dup_count}"

def test_liked_column_values(real_raw_data_path):
    import pandas as pd
    df = pd.read_csv(real_raw_data_path, sep='\t')
    df.columns = df.columns.str.strip()
    assert df['Liked'].isin([0, 1]).all(), "Found non-binary or invalid values in 'Liked' column"


