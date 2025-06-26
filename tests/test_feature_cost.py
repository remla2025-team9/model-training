import time
import tracemalloc
import joblib
import pandas as pd
import pytest
from src import config  

@pytest.fixture
def sample_dataset():
   
    data = {
        "Review": [
            "I love this product",
            "Terrible experience, will not buy again",
            "Pretty good, could be better",
        ],
        "Liked": [1, 0, 1],
    }
    return pd.DataFrame(data)


def test_prediction_latency(sample_dataset):
  
    vectorizer = joblib.load(config.VECTORIZERS_DIR / "vectorizer.joblib")
    model = joblib.load(config.MODELS_DIR / "model.joblib")

    texts = sample_dataset["Review"].tolist()
    X = vectorizer.transform(texts)

    start = time.time()
    model.predict(X)
    elapsed = time.time() - start

    assert elapsed < 5, f"Prediction too slow: {elapsed:.4f}s"


def test_memory_usage(sample_dataset):
    tracemalloc.start()

    vectorizer = joblib.load(config.VECTORIZERS_DIR / "vectorizer.joblib")
    model = joblib.load(config.MODELS_DIR / "model.joblib")

    texts = sample_dataset["Review"].tolist()
    X = vectorizer.transform(texts)
    model.predict(X)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    assert peak_mb < 50, f"Peak memory usage too high: {peak_mb:.2f} MB"
