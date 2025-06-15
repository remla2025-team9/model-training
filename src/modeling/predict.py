"""Prediction routines and evaluation reporting."""

import argparse
import logging
import joblib
import pandas as pd
import json
from scipy.sparse import load_npz
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import os
from .. import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Evaluate a trained sentiment classifier on test data and save metrics.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test features from %s", config.TEST_FEATURES_FILE)
    X_test_features = load_npz(config.TEST_FEATURES_FILE)

    logger.info("Loading test labels from %s", config.TEST_LABELS_FILE)
    y_test = pd.read_csv(config.TEST_LABELS_FILE)["label"].values

    logger.info("Loading trained classifier from %s", args.model_path)
    if not os.path.exists(args.model_path):
        logger.error("Model file not found at: %s", args.model_path)
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")
    classifier = joblib.load(args.model_path)

    logger.info("Making predictions on test data...")
    try:
        x_test_dense = X_test_features.toarray()
    except MemoryError:
        logger.error("MemoryError converting sparse test features to dense.")
        raise

    y_pred = classifier.predict(x_test_dense)

    logger.info("Calculating evaluation metrics...")
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred, zero_division=0))
    logger.info("Confusion Matrix:\n%s", cm)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    with open(config.EVALUATION_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Evaluation metrics saved to %s", config.EVALUATION_METRICS_FILE)


if __name__ == "__main__":
    # Entry point for the sentiment classifier evaluation script.
    parser = argparse.ArgumentParser(
        description="Evaluate a trained sentiment analysis classifier."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained classifier (.joblib file).",
    )
    parsed_args = parser.parse_args()

    if not (config.TEST_FEATURES_FILE.exists() and config.TEST_LABELS_FILE.exists()):
        logger.warning(
            "Processed test data not found. Please run 'python -m sentiment_src.dataset' first."
        )
    elif not os.path.exists(parsed_args.model_path):
        logger.error("Model file for evaluation not found at: %s", parsed_args.model_path)
    else:
        main(parsed_args)
