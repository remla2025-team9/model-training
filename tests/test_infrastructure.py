import subprocess
import json
from src import config

def test_dvc_pipeline_produces_valid_model_and_metrics():
    """
    Integration Test: Run the DVC pipeline and verify it produces a valid model and metrics.
    """

    # Dynamically find the first .joblib model in the models directory
    joblib_files = list(config.MODELS_DIR.glob("*.joblib"))
    assert joblib_files, f"No .joblib model files found in {config.MODELS_DIR}"
    model_path = joblib_files[0]

    metrics_file = config.EVALUATION_METRICS_FILE

    try:
        # Run the full DVC pipeline
        result = subprocess.run(["dvc", "repro", "--force"], capture_output=True, text=True)
        assert result.returncode == 0, f"DVC pipeline failed:\n{result.stderr}"

        # Verify model exists
        assert model_path.exists(), f"Trained model not found at {model_path}"

        # Verify evaluation metrics file exists and is valid
        assert metrics_file.exists(), f"Evaluation metrics file not found at {metrics_file}"

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        # Check accuracy
        accuracy = metrics.get("accuracy", 0)
        assert accuracy > 0.6, f"Accuracy too low: {accuracy:.2f}"

    finally:
        # Optional: Clean up generated files if needed
        if model_path.exists():
            model_path.unlink()
        if metrics_file.exists():
            metrics_file.unlink()
