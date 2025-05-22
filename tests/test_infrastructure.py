import pytest
import joblib
import os
from pathlib import Path
from sklearn.base import BaseEstimator
from src import config
from src.modeling import train

def test_ml_pipeline_train_and_save_model(tmp_path):
   
    if not (config.TRAIN_FEATURES_FILE.exists() and config.TRAIN_LABELS_FILE.exists()):
        pytest.skip("Training data missing, skipping full pipeline integration test.")

  
    class Args:
        model_type = 'nb'
        model_version = 'test'

    args = Args()

   
    original_models_dir = config.MODELS_DIR
    config.MODELS_DIR = tmp_path

    try:
       
        train.main(args)

        expected_model_file = tmp_path / f"sentiment_classifier-{args.model_type}-v{args.model_version}.joblib"
        assert expected_model_file.exists(), "Model file was not saved."

       
        model = joblib.load(expected_model_file)

        assert isinstance(model, BaseEstimator), "Loaded model is not a scikit-learn estimator."

    finally:

        config.MODELS_DIR = original_models_dir
