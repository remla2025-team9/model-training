# Restaurant Sentiment Analysis - Model Training

This repository contains scripts to train, evaluate, and export a sentiment analysis model for restaurant reviews. The project follows a structured layout inspired by Cookiecutter Data Science.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="Cookiecutter Data Science Template Badge" />
</a>

---

## Project Organization

The project follows a structure adapted from the Cookiecutter Data Science template:

```
├── .github/             # GitHub Actions workflows (CI/CD - Planned)
│   └── workflows/
├── data/
│   ├── raw/             # Original, immutable data (e.g., training_data.tsv)
│   └── processed/       # Processed data ready for modeling (features & labels)
├── docs/                # Project documentation (e.g., MkDocs)
├── models/              # Trained model artifacts (e.g., sentiment_classifier-nb-v1.0.0.joblib)
├── notebooks/           # Jupyter notebooks for exploration
├── reports/             # Evaluation metrics, figures
│   └── evaluation_metrics.json
├── src/       # Python source code for the pipeline
│   ├── __init__.py
│   ├── config.py        # Configuration variables (paths, defaults)
│   ├── dataset.py       # Data loading, preprocessing (feature generation), splitting
│   └── modeling/
│       ├── __init__.py
│       ├── train.py     # Model training script
│       └── predict.py   # Model evaluation script
├── .gitignore           # Files for Git to ignore
├── pyproject.toml       # Project metadata and tool configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Prerequisites

*   Python 3.8 or newer
*   Git

**Installation:**

1.  **Install Python dependencies:**
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    The `requirements.txt` should include your custom preprocessing library:
    ```
    # In requirements.txt
    # ... other packages ...
    git+https://github.com/remla2025-team9/lib-ml.git # Or your specific preprocessing lib
    ```

---

## Usage - Running the Pipeline Manually

Currently, the pipeline stages are run as individual Python scripts in sequence. DVC integration for automated pipeline management is planned for a future update.

**Step 1: Data Processing (Generate features & split data)**

This script loads the raw data, uses an external library for text preprocessing to generate features, splits the data into training and test sets, and saves the processed features and labels.

```bash
python -m src.dataset --test_size 0.2
```
*   **Inputs:** Reads from `data/raw/training_data.tsv` (or as specified by `--raw_data_path` argument or `src/config.py`).
*   **Outputs:**
    *   `data/processed/train_features.npz`
    *   `data/processed/train_labels.csv`
    *   `data/processed/test_features.npz`
    *   `data/processed/test_labels.csv`

**Step 2: Model Training**

This script loads the processed training features and labels, trains a classifier, and saves the trained model.

```bash
python -m src.modeling.train --model_type logistic --model_version 1.0.0
```
*   **Inputs:** Reads `data/processed/train_features.npz` and `data/processed/train_labels.csv`.
*   **Parameters:**
    *   `--model_type`: Choose 'nb' (Gaussian Naive Bayes) or 'logistic' (Logistic Regression).
    *   `--model_version`: Specify a version for the output model file (e.g., '1.0.0').
*   **Outputs:** Saves the trained model to `models/sentiment_classifier-<model_type>-v<model_version>.joblib` (e.g., `models/sentiment_classifier-logistic-v1.0.0.joblib`).

**Step 3: Model Evaluation**

This script loads the processed test features and labels, loads a trained model, makes predictions, and saves evaluation metrics.

```bash
python -m src.modeling.predict --model_path models/sentiment_classifier-logistic-v1.0.0.joblib
```
*   **Inputs:**
    *   Reads `data/processed/test_features.npz` and `data/processed/test_labels.csv`.
    *   `--model_path`: Path to the trained model file saved in Step 2.
*   **Outputs:**
    *   Prints classification report and confusion matrix to the console.
    *   Saves detailed metrics to `reports/evaluation_metrics.json`.

---

## Parameters

Key parameters for the scripts (e.g., test split ratio, model type) can be passed as command-line arguments as shown above. Default file paths are managed in `src/config.py`.
*(A `params.yaml` file is included for planned DVC integration, where these parameters will be centrally managed.)*

---

## CI/CD

On every Git tag (e.g. `v1.0.0`), the `.github/workflows/train-release.yaml` workflow will:

1. Checkout the code
2. Install dependencies
3. Run `preprocess.py`
4. Run `train.py` or `improved_train.py`
5. Upload `models/model-v<tag>.pkl` as a GitHub Release asset

The public download URL will be:

```
https://github.com/remla2025-team9/model-training/releases/download/vx.y.z/sentiment_pipeline-vx.y.z.joblib
```

## Model integration
The trained model can now be integrated and used for predictions

### 1. From a GitHub Release
Each stable version tag `vX.Y.Z` has its model attached as a release asset:
```
https://github.com/remla2025-team9/model-training/releases/download/vX.Y.Z/sentiment_pipeline-vX.Y.Z.joblib
```
In your code or service, set the URL and fetch at startup:
```python
import os
import pathlib
import requests
import joblib

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/remla2025-team9/model-training/releases/download/v1.2.3/"
    "sentiment_pipeline-v1.2.3.joblib"
)
```
Download and cache the model:
```python
cache_dir = pathlib.Path("/cache/models")
cache_dir.mkdir(parents=True, exist_ok=True)
local_path = cache_dir / pathlib.Path(MODEL_URL).name

if not local_path.exists():
    resp = requests.get(MODEL_URL)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
```
Load the model:
```python
model = joblib.load(local_path)
```
### 2. From a local file
It is also possible to load the model from a local file, either by cloning the repo and following the steps above to train and generate the model, or by downloading the model file directly from the GitHub release page.

## How to Run Tests with Coverage Reporting

You can run the test suite with coverage measurement using the following command:

```bash
pytest --cov=src tests/
```