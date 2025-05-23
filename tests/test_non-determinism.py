import pytest
import joblib
import numpy as np
from src.modeling.train import main as train_main
from src.modeling.predict import main as predict_main
from argparse import Namespace
from pathlib import Path
import json
from src import config

def get_latest_model_file(models_dir: Path):
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return model_files[0]

@pytest.fixture(scope="module")
def base_model_score():
    train_args = Namespace(model_type='logistic', model_version='stablebase')
    train_main(train_args)

    latest_model = get_latest_model_file(config.MODELS_DIR)
    predict_args = Namespace(model_path=str(latest_model))
    predict_main(predict_args)

    with open(config.EVALUATION_METRICS_FILE) as f:
        metrics = json.load(f)
    return metrics['accuracy']

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_model_accuracy_stability(seed, base_model_score):
    """
    Test the stability of model accuracy across different random seeds,
    using the latest model generated each time.
    """
    version_str = f"stabletest-{seed}"
    train_args = Namespace(model_type='logistic', model_version=version_str)
    train_main(train_args)

    latest_model = get_latest_model_file(config.MODELS_DIR)
    predict_args = Namespace(model_path=str(latest_model))
    predict_main(predict_args)

    with open(config.EVALUATION_METRICS_FILE) as f:
        metrics = json.load(f)
    new_score = metrics['accuracy']

    assert abs(new_score - base_model_score) <= 0.03, f"Score unstable with seed={seed}: {new_score} vs {base_model_score}"
