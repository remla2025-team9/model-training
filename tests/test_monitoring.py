import subprocess
import time
import json
import psutil
from pathlib import Path
from src import config

def measure_process_memory(pid):
    try:
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        return mem_info.rss / (1024 * 1024)
    except psutil.NoSuchProcess:
        return None

def test_dvc_pipeline_performance():
    """
    Run the full DVC pipeline and monitor its runtime duration and memory usage.
    Assert that performance stays within reasonable limits.
    """

    # Dynamically find the latest model and use that for checking
    model_files = sorted(config.MODELS_DIR.glob("*.joblib"), key=lambda f: f.stat().st_mtime, reverse=True)
    metrics_file = config.EVALUATION_METRICS_FILE

    # === Step 1: Run DVC pipeline with monitoring ===
    start_time = time.time()
    proc = subprocess.Popen(["dvc", "repro", "--force"])
    
    max_memory = 0
    while proc.poll() is None:
        mem = measure_process_memory(proc.pid)
        if mem is not None and mem > max_memory:
            max_memory = mem
        time.sleep(1)

    end_time = time.time()
    duration = end_time - start_time
    returncode = proc.returncode

    # === Step 2: Check success ===
    assert returncode == 0, f"DVC pipeline failed with return code {returncode}"

    # === Step 3: Check model file exists ===
    assert model_files, f"No .joblib model found in {config.MODELS_DIR}"
    model_path = model_files[0]
    assert model_path.exists(), f"Model file not found: {model_path}"

    # === Step 4: Check metrics file exists and contains accuracy > 0.6 ===
    assert metrics_file.exists(), f"Evaluation metrics file not found: {metrics_file}"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    accuracy = metrics.get("accuracy", 0)
    assert accuracy > 0.6, f"Accuracy too low: {accuracy:.2f}"

    # === Step 5: Performance thresholds ===
    MAX_TIME = 300        # 5 minutes
    MAX_MEMORY = 700      # MB (adjust depending on your machine/model size)

    assert duration < MAX_TIME, f"DVC pipeline too slow: {duration:.2f}s"
    assert max_memory < MAX_MEMORY, f"DVC pipeline memory too high: {max_memory:.2f}MB"
