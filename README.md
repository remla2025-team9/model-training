# Restaurant Sentiment Analysis - Model Training

This repository contains scripts to train, evaluate, and export a sentiment analysis model for restaurant reviews. The project follows a structured layout inspired by Cookiecutter Data Science.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="Cookiecutter Data Science Template Badge" />
</a>

---

## Project Organization

The project follows a structure adapted from the Cookiecutter Data Science template:

```
├── .github/             # GitHub Actions workflows for CI/CD
│   └── workflows/
│       ├── integration.yml   # Code quality and pipeline validation
│       ├── testing.yml       # Test execution and coverage reporting
│       ├── delivery.yml      # Automated pre-release creation
│       └── deployment.yml    # Stable release deployment
├── data/
│   ├── raw/             # Original, immutable data
│   └── processed/       # Processed data ready for modeling (features & labels)
├── docs/                # Project documentation
├── linting/             # Code quality configuration
├── models/              # Trained model artifacts
├── notebooks/           # Jupyter notebooks for exploration
├── reports/             # Evaluation metrics and figures
├── src/                 # Python source code for the pipeline
│   ├── __init__.py
│   ├── config.py        # Configuration variables (paths, defaults)
│   ├── dataset.py       # Data loading routines
│   ├── features.py      # Feature generation and preprocessing
│   └── modeling/
│       ├── __init__.py
│       ├── train.py     # Model training script
│       └── predict.py   # Model evaluation script
├── tests/               # Test files
├── vectorizers/         # Saved vectorizers and preprocessing components
├── dvc.yaml             # DVC pipeline configuration
├── dvc.lock             # DVC pipeline lock file
├── params.yaml          # Pipeline parameters
├── pyproject.toml       # Project metadata and tool configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Setup

### Prerequisites
*   Python 3.8 or newer
*   Git
*   AWS CLI (for DVC remote storage)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd model-training
   ```

2. **Set up Python virtual environment:**
   ```bash
   python -m venv .venv

   # On MacOS and Linux:
   source .venv/bin/activate  
   
   # On Windows: 
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials (for DVC):**
   ```bash
   pip install awscli
   aws configure  # Set your AWS Access Key, Secret Key, and default region
   ```

5. **Initialize DVC:**
   ```bash
   dvc pull  # Download data and model files from remote storage
   ```

---

## Usage

### Option 1: Run Complete Pipeline with DVC (Recommended)

The DVC pipeline manages data versioning and ensures reproducible model training:

```bash
# Run the entire pipeline
dvc repro

# Or run specific stages
dvc repro load_data    # Download raw data
dvc repro prepare     # Generate features
dvc repro train       # Train model
dvc repro evaluate    # Evaluate model
```

#### Pipeline Stages

The pipeline consists of four sequential stages:

1. **`load_data`** - Downloads raw training data from Google Drive
   - **Outputs:** `data/raw/training_data.tsv`

2. **`prepare`** - Processes raw data into ML-ready features
   - Preprocesses text using external Preprocessor
   - Applies CountVectorizer for feature extraction
   - Splits data into train/test sets
   - **Outputs:** processed features, labels, and fitted vectorizer

3. **`train`** - Trains machine learning model
   - Supports Logistic Regression and Naive Bayes
   - **Outputs:** `models/model.joblib`

4. **`evaluate`** - Evaluates model performance
   - Generates accuracy, precision, recall, F1 metrics
   - **Outputs:** `reports/evaluation_metrics.json`

### Option 2: Run Pipeline Manually

For development or debugging, you can run individual steps:

**Step 1: Load Data**

Download raw training data from Google Drive:

```bash
python -m src.dataset
```
*   **Outputs:** `data/raw/training_data.tsv`

**Step 2: Data Processing (Generate features & split data)**

Process raw data into features and labels ready for modeling:

```bash
python -m src.features
```
*   **Inputs:** `data/raw/training_data.tsv`
*   **Outputs:**
    *   `data/processed/train_features.npz`
    *   `data/processed/train_labels.csv`
    *   `data/processed/test_features.npz`
    *   `data/processed/test_labels.csv`
    *   `vectorizers/vectorizer.joblib`

**Step 3: Model Training**

This script loads the processed training features and labels, trains a classifier, and saves the trained model.

```bash
python -m src.modeling.train --model_type logistic
```
*   **Inputs:** Reads `data/processed/train_features.npz` and `data/processed/train_labels.csv`.
*   **Parameters:**
    *   `--model_type`: Choose 'nb' (Gaussian Naive Bayes) or 'logistic' (Logistic Regression).
*   **Outputs:** Saves the trained model to `models/model.joblib`

**Step 4: Model Evaluation**

This script loads the processed test features and labels, loads a trained model, makes predictions, and saves evaluation metrics.

```bash
python -m src.modeling.predict --model_path models/model.joblib
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

## CI/CD Workflows

This project implements a complete CI/CD pipeline with four automated workflows for quality assurance and model deployment.

### Integration (`integration.yml`)
**Trigger:** Pull requests to `main` branch  
**Purpose:** Validates code quality and pipeline integrity

- Runs code quality checks (pylint, flake8, bandit)
- Executes full DVC pipeline to ensure reproducibility
- Prevents breaking changes from being merged

### Testing (`testing.yml`) 
**Trigger:** Push and pull requests to `main` branch  
**Purpose:** Comprehensive testing and metrics reporting

- Runs unit and integration tests with coverage reporting
- Generates pylint score badges and security analysis
- Auto-updates README with test results and metrics
- Commits updated badges and reports

### Delivery (`delivery.yml`)
**Trigger:** Every push to `main` branch  
**Purpose:** Automated pre-release creation

- Trains model with latest code and data
- Creates pre-release tags with semantic versioning
- Uploads model, vectorizer, and metrics as release assets
- Enables testing of new model versions

### Deployment (`deployment.yml`)
**Trigger:** Manual workflow dispatch  
**Purpose:** Stable release deployment

- Allows selection of release level (patch/minor/major)
- Creates stable releases with trained models and vectorizers
- Automatically prepares next development cycle
- Handles version tagging and Git operations


## Model Integration

The trained sentiment analysis model and its vectorizer can be integrated into applications for making predictions. Both components are packaged together in each release.

### Download from GitHub Release

Each stable version tag `{{ MODEL_RELEASE_VERSION }}` includes both the trained model and vectorizer as release assets:

```python
import os
import pathlib
import requests
import joblib

# Set release version and download URLs
MODEL_RELEASE_VERSION = os.getenv("MODEL_RELEASE_VERSION", "v1.0.0")
BASE_URL = f"https://github.com/remla2025-team9/model-training/releases/download/{MODEL_RELEASE_VERSION}"

MODEL_URL = f"{BASE_URL}/model.joblib"
VECTORIZER_URL = f"{BASE_URL}/vectorizer.joblib"

def download_and_cache_file(url, cache_dir, filename):
    """Download and cache a file locally."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename
    
    if not local_path.exists():
        resp = requests.get(url)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Using cached {filename}")
    
    return local_path

# Download both model and vectorizer
cache_dir = pathlib.Path("/cache/models")
model_path = download_and_cache_file(MODEL_URL, cache_dir, "model.joblib")
vectorizer_path = download_and_cache_file(VECTORIZER_URL, cache_dir, "vectorizer.joblib")

# Load the components
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
```

### Making Predictions

To use the model for sentiment analysis, you need the preprocessor, vectorizer, and trained model:

```python
from sentiment_analysis_preprocessing.preprocesser import Preprocessor

def predict_sentiment(text_review, model, vectorizer):
    """
    Predict sentiment for a restaurant review.
    
    Args:
        text_review (str): The review text to analyze
        model: The trained sentiment classifier
        vectorizer: The fitted CountVectorizer from training
    
    Returns:
        dict: Prediction results with label and confidence
    """
    # Step 1: Preprocess text using the same preprocessor from training
    preprocessor = Preprocessor()
    preprocessed_text = preprocessor.transform([text_review])
    
    # Step 2: Transform preprocessed text using the vectorizer
    text_features = vectorizer.transform(preprocessed_text)
    
    # Step 3: Make prediction
    prediction = model.predict(text_features)[0]
    confidence = max(model.predict_proba(text_features)[0])
    
    return {
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": round(confidence, 3)
    }

# Example usage
review_text = "The food was absolutely delicious and the service was excellent!"
result = predict_sentiment(review_text, model, vectorizer)
print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
```

### Integration Notes

- **Complete pipeline**: Always use Preprocessor → CountVectorizer → Model in that exact order
- **Version consistency**: Download model and vectorizer from the same release to ensure compatibility
- **Preprocessing requirement**: The Preprocessor step is essential and must match the training pipeline
- **Performance**: Consider caching the loaded components to avoid reloading for multiple predictions
- **Error handling**: Add appropriate error handling for network requests and file operations in production use



## How to Run Tests with Coverage Reporting

You can run the test suite with coverage measurement using the following command:

```bash
pytest --cov=src tests/
```


## Code Quality Checks

We use three tools to ensure code quality: Pylint, Flake8, and Bandit.

### Pylint

To run pylint (checking  naming, code smells, and structure, includes a custom plugin to detect hardcoded seeds)

```bash
pip install pylint
pylint --rcfile=linting/.pylintrc src/
```

### Flake8

To run flake8 (PEP8 formatting)
```bash
pip install flake8
flake8 --config=linting/.flake8 src/
```

### Bandit

To run bandit

```bash
pip install bandit
bandit -r src/ -c linting/bandit.yaml
```



## Coverage Report

<!-- coverage-start -->

| File                       | Stmts | Miss | Cover |
| -------------------------- | ----- | ---- | ----- |
| `src/__init__.py`          | 1     | 0    | 100%  |
| `src/config.py`            | 13    | 0    | 100%  |
| `src/dataset.py`           | 60    | 4    | 93%   |
| `src/features.py`          | 22    | 0    | 100%  |
| `src/modeling/__init__.py` | 0     | 0    | 100%  |
| `src/modeling/predict.py`  | 47    | 2    | 96%   |
| `src/modeling/train.py`    | 40    | 1    | 98%   |
| `src/plots.py`             | 0     | 0    | 100%  |
| **Total**                  | 183   | 7    | 96%   |

<!-- coverage-end -->

Generated using `pytest` and `pytest-cov`:

| Module Path                | Stmts   | Miss  | Cover   |
| -------------------------- | ------- | ----- | ------- |
| `src/__init__.py`          | 1       | 0     | 100%    |
| `src/config.py`            | 13      | 0     | 100%    |
| `src/dataset.py`           | 60      | 4     | 93%     |
| `src/features.py`          | 22      | 22    | 0%      |
| `src/modeling/__init__.py` | 0       | 0     | 100%    |
| `src/modeling/predict.py`  | 47      | 2     | 96%     |
| `src/modeling/train.py`    | 40      | 1     | 98%     |
| `src/plots.py`             | 0       | 0     | 100%    |
| **Total**                  | **161** | **7** | **96%** |

### Test Summary

- **All 28 test cases passed**
- **Overall coverage: 96%**

## 🧪 Test Adequacy Metrics
<!-- adequacy-start -->
| Metric               | Value  |
| -------------------- | ------ |
| Accuracy             | 0.7556 |
| Precision (weighted) | 0.7563 |
| Recall (weighted)    | 0.7556 |
| F1 Score (weighted)  | 0.7536 |

<details>
<summary>Confusion Matrix</summary>
[[55, 27], [17, 81]]
</details>
<!-- adequacy-end -->

<!-- BANDIT_START -->
## Bandit Security Analysis

| File                       | LOC | High | Medium | Low |
| -------------------------- | --- | ---- | ------ | --- |
| `src/__init__.py`          | 2   | 0    | 0      | 0   |
| `src/config.py`            | 14  | 0    | 0      | 0   |
| `src/dataset.py`           | 115 | 0    | 0      | 0   |
| `src/features.py`          | 27  | 0    | 0      | 0   |
| `src/modeling/__init__.py` | 0   | 0    | 0      | 0   |
| `src/modeling/predict.py`  | 83  | 0    | 0      | 0   |
| `src/modeling/train.py`    | 60  | 0    | 0      | 0   |
| `src/plots.py`             | 0   | 0    | 0      | 0   |
<!-- BANDIT_END -->

## Linting Scores
![Pylint Score](linting/pylint_score.svg)