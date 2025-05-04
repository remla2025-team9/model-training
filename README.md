# Sentiment Analysis Model Training

This repository contains tools and scripts to train, evaluate, and export a sentiment analysis model in a decoupled, versioned, and reproducible manner.

---

## Repository Structure

```text
.
├── data/                   # Raw datasets
│   └── training_data.tsv   # Training dataset
├── scripts/                # Training source code
│   ├── preprocess.py       # (Uses lib-ml) Raw → features → .npz
│   └── train.py            # End-to-end pipeline trainer on TSV
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .github/workflows/      # CI/CD workflows
│   ├── train-release.yaml  # Automate train & GitHub Release on tag\
│   └── CODEOWNERS

```

---

## Prerequisites

* Python 3.8 or newer
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
* Install your preprocessing library:

  ```bash
  pip install git+https://github.com/remla2025-team9/lib-ml.git
  ```
---

## Usage
Standalone preprocess + train + eval on a given dataset :

```bash
python src/train.py \
  --train-input path/to/training/data.tsv \
  --model-output destination/path/of/model.joblib \
  --version x.x.x \
  --model-type [dn|logistic] \
```

Both scripts output:

* A classification report (printed to console)
* A confusion matrix (printed to console)
* Saved model artifact (`.joblib`)

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
It is also possible to load the model from a local file, either by cloning the repo and following the steps above to train and generate the model, or by downloading the model file directly from the GitHub release page. For quick inspection, open the Actions tab, select the workflow run that built the model, and download the .joblib file under Artifacts.