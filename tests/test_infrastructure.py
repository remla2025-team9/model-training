import subprocess
import json
from pathlib import Path
from src import config

def test_training_and_prediction_pipeline(tmp_path):
    """
    Integration Test: Verify the complete training + prediction pipeline runs successfully 
    and achieves a basic level of accuracy.
    """

    model_type = "logistic"
    model_version = "1.0.0"
    model_filename = f"sentiment_classifier-{model_type}-v{model_version}.joblib"
    model_path = config.MODELS_DIR / model_filename

    # Step 1: Run the training script
    train_cmd = [
        "python", "-m", "src.modeling.train",
        "--model_type", model_type,
        "--model_version", model_version,
    ]
    result_train = subprocess.run(train_cmd, capture_output=True, text=True)
    assert result_train.returncode == 0, f"Training failed:\n{result_train.stderr}"

    # Step 2: Verify the model file was successfully created
    assert model_path.exists(), f"Trained model not found at {model_path}"

    # Step 3: Run the prediction script
    predict_cmd = [
        "python", "-m", "src.modeling.predict",
        "--model_path", str(model_path)
    ]
    result_predict = subprocess.run(predict_cmd, capture_output=True, text=True)
    assert result_predict.returncode == 0, f"Prediction failed:\n{result_predict.stderr}"

    # Step 4: Check that the evaluation metrics file exists and load it
    metrics_file = config.EVALUATION_METRICS_FILE
    assert metrics_file.exists(), f"Evaluation metrics file not found at {metrics_file}"

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Step 5: Ensure the accuracy meets a minimum threshold
    accuracy = metrics["accuracy"]
    assert accuracy > 0.7, f"Accuracy too low: {accuracy:.2f}"
