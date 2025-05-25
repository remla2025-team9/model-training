import time
import joblib
from src import config
from scipy.sparse import load_npz

def load_latest_model():
    """
    Load the most recently modified model file (*.joblib) from the models directory.
    """
    model_files = list(config.MODELS_DIR.glob("*.joblib"))
    model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return joblib.load(model_files[0])

def prepare_features_for_model(model, X):
    """
    Decide whether to convert a sparse matrix to a dense matrix based on the model type.
    Currently, conversion is done for GaussianNB only; other models are assumed to support sparse matrices.
    """
    
    from sklearn.naive_bayes import GaussianNB

    if isinstance(model, GaussianNB):
        if hasattr(X, "toarray"):
            return X.toarray()
    return X

def test_prediction_latency():
    """
    Test the latency of the model's prediction on the test feature dataset.
    """
    model = load_latest_model()
    X_test = load_npz(config.TEST_FEATURES_FILE)
    X_test = prepare_features_for_model(model, X_test)

    start = time.time()
    model.predict(X_test)
    end = time.time()

    latency = end - start
    print(f"Prediction latency: {latency:.4f} seconds")
    assert latency < 1.0, "Prediction latency too high"

def test_memory_usage():
    """
    Test the peak memory usage during model prediction on the test dataset.
    """
    import tracemalloc
    model = load_latest_model()
    X_test = load_npz(config.TEST_FEATURES_FILE)
    X_test = prepare_features_for_model(model, X_test)

    tracemalloc.start()
    model.predict(X_test)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Memory usage during prediction: Current={current/1024:.2f}KB, Peak={peak/1024:.2f}KB")
    assert peak < 100 * 1024 * 1024, "Memory usage too high"
