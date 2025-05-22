import sys
from pathlib import Path
import random
import numpy as np
import pytest
import json
from src.modeling import train, predict
import src.config as config

NUM_RUNS = 3

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

@pytest.mark.parametrize("run", range(NUM_RUNS))
def test_model_non_determinism(run):
    # Skip test if processed train/test feature or label files are missing
    if not (config.TRAIN_FEATURES_FILE.exists() and config.TRAIN_LABELS_FILE.exists() and
            config.TEST_FEATURES_FILE.exists() and config.TEST_LABELS_FILE.exists()):
        pytest.skip("Processed train or test feature files missing, skipping non-determinism test.")

    seed = 42 + run
    set_seeds(seed)

    class Args:
        model_type = 'logistic'
        model_version = f"test_run_{run}"

    train.main(Args())

    version_str = Args.model_version.lstrip('v')
    model_filename = f"sentiment_classifier-{Args.model_type}-v{version_str}.joblib"
    model_path = config.MODELS_DIR / model_filename

    assert model_path.exists(), f"Model file not found: {model_path}"

    class PredictArgs:
        model_path = str(model_path)

    predict.main(PredictArgs())

    metrics_file = config.EVALUATION_METRICS_FILE
    assert metrics_file.exists(), "Evaluation metrics file missing"

    with open(metrics_file) as f:
        metrics = json.load(f)

    accuracy = metrics.get('accuracy', 0)
    print(f"Run {run}, accuracy: {accuracy}")

    assert 0.5 <= accuracy <= 1.0, f"Run {run} accuracy out of expected range: {accuracy}"
