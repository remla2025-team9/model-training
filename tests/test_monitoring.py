import numpy as np
import pandas as pd
import pytest
from src import config

def test_train_test_feature_shape_consistency():
    """
    Test consistency of feature dimensions between training and test datasets.
    """
    if not (config.TRAIN_FEATURES_FILE.exists() and config.TEST_FEATURES_FILE.exists()):
        pytest.skip("Feature files missing, skipping shape consistency test.")

    train_data = np.load(config.TRAIN_FEATURES_FILE)["arr_0"]
    test_data = np.load(config.TEST_FEATURES_FILE)["arr_0"]

    assert train_data.shape[1] == test_data.shape[1], (
        f"Feature dimension mismatch: train {train_data.shape[1]}, test {test_data.shape[1]}"
    )

def test_feature_label_alignment():
    """
    Test that the number of training samples matches the number of labels.
    """
    if not (config.TRAIN_FEATURES_FILE.exists() and config.TRAIN_LABELS_FILE.exists()):
        pytest.skip("Training feature or label file missing, skipping alignment test.")

    train_features = np.load(config.TRAIN_FEATURES_FILE)["arr_0"]
    train_labels = pd.read_csv(config.TRAIN_LABELS_FILE)

    assert len(train_features) == len(train_labels), (
        f"Mismatch between number of training samples and labels: "
        f"{len(train_features)} features vs {len(train_labels)} labels"
    )
