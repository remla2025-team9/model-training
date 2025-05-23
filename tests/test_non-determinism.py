import pytest
import joblib
import numpy as np
from src.modeling.train import main as train_main
from src.modeling.predict import main as predict_main
from argparse import Namespace
from pathlib import Path
import json
from src import config

@pytest.fixture(scope="module")
def base_model_score():
   
    train_args = Namespace(model_type='logistic', model_version='1.0.0')
    train_main(train_args)

    model_path = config.MODELS_DIR / "sentiment_classifier-logistic-v1.0.0.joblib"
    predict_args = Namespace(model_path=str(model_path))
    predict_main(predict_args)

    with open(config.EVALUATION_METRICS_FILE) as f:
        metrics = json.load(f)
    return metrics['accuracy']  

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_model_accuracy_stability(seed, base_model_score):
    """
    Test the stability of model accuracy across different random seeds.
    """
    version_str = f"stabletest-{seed}"
    train_args = Namespace(model_type='logistic', model_version=version_str)
    train_main(train_args)

    model_path = config.MODELS_DIR / f"sentiment_classifier-logistic-v{version_str}.joblib"
    predict_args = Namespace(model_path=str(model_path))
    predict_main(predict_args)

    with open(config.EVALUATION_METRICS_FILE) as f:
        metrics = json.load(f)
    new_score = metrics['accuracy']

    assert abs(new_score - base_model_score) <= 0.03, f"Score unstable with seed={seed}: {new_score} vs {base_model_score}"
