import subprocess
import time
import json
import psutil
from pathlib import Path
from src import config  

def measure_process_memory(pid):
    """
    Measure the resident set size (RSS) memory usage of a process in MB.
    Returns None if the process does not exist.
    """
    try:
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        return mem_info.rss / (1024 * 1024)  
    except psutil.NoSuchProcess:
        return None

def test_training_and_prediction_performance():
    """
    Monitor runtime duration and peak memory usage,
    and assert that these metrics stay within predefined acceptable thresholds.
    """
    model_type = "logistic"
    model_version = "1.0.0"
    model_filename = f"sentiment_classifier-{model_type}-v{model_version}.joblib"
    model_path = config.MODELS_DIR / model_filename
    
    # === Step 1: Measure training time and memory usage ===
    train_cmd = [
        "python", "-m", "src.modeling.train",
        "--model_type", model_type,
        "--model_version", model_version,
    ]
    start_train = time.time()
    proc_train = subprocess.Popen(train_cmd)
    
    # Monitor memory usage during training process every 1 second (simplified)
    max_train_mem = 0
    while proc_train.poll() is None:
        mem = measure_process_memory(proc_train.pid)
        if mem is not None and mem > max_train_mem:
            max_train_mem = mem
        time.sleep(1)
    end_train = time.time()
    train_duration = end_train - start_train
    train_returncode = proc_train.returncode
    
    # Assert training process exited successfully
    assert train_returncode == 0, "Training process failed"
    
    
    # === Step 2: Measure prediction time and memory usage ===
    predict_cmd = [
        "python", "-m", "src.modeling.predict",
        "--model_path", str(model_path)
    ]
    start_pred = time.time()
    proc_pred = subprocess.Popen(predict_cmd)
    
    # Monitor memory usage during prediction process every 1 second (simplified)
    max_pred_mem = 0
    while proc_pred.poll() is None:
        mem = measure_process_memory(proc_pred.pid)
        if mem is not None and mem > max_pred_mem:
            max_pred_mem = mem
        time.sleep(1)
    end_pred = time.time()
    pred_duration = end_pred - start_pred
    pred_returncode = proc_pred.returncode
    
    # Assert prediction process exited successfully
    assert pred_returncode == 0, "Prediction process failed"
    
    
    # === Step 3: Set reasonable thresholds to prevent sudden slowdown or memory spikes ===
    MAX_TRAIN_TIME = 300   # 5 minutes
    MAX_PRED_TIME = 30     # 30 seconds
    MAX_TRAIN_MEM = 500    # MB
    MAX_PRED_MEM = 300     # MB
    
    assert train_duration < MAX_TRAIN_TIME, f"Training too slow: {train_duration:.2f}s"
    assert pred_duration < MAX_PRED_TIME, f"Prediction too slow: {pred_duration:.2f}s"
    assert max_train_mem < MAX_TRAIN_MEM, f"Training memory too high: {max_train_mem:.2f}MB"
    assert max_pred_mem < MAX_PRED_MEM, f"Prediction memory too high: {max_pred_mem:.2f}MB"
