"""Configuration parameters and default file paths."""
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Model paths
MODELS_DIR = ROOT_DIR / "models"

# Reports paths
REPORTS_DIR = ROOT_DIR / "reports"

DEFAULT_RAW_DATA_FILE = RAW_DATA_DIR / "training_data.tsv"
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.npz"
TRAIN_LABELS_FILE = PROCESSED_DATA_DIR / "train_labels.csv"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "test_features.npz"
TEST_LABELS_FILE = PROCESSED_DATA_DIR / "test_labels.csv"
EVALUATION_METRICS_FILE = REPORTS_DIR / "evaluation_metrics.json"

VECTORIZERS_DIR = ROOT_DIR / "vectorizers" / "preprocessor.joblib"
