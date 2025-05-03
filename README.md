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
