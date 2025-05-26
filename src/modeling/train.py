"""Training routines for the sentiment classification model."""

import argparse
import logging
import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import os
from .. import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading training features from {config.TRAIN_FEATURES_FILE}")
    X_train_features = load_npz(config.TRAIN_FEATURES_FILE)
    
    logger.info(f"Loading training labels from {config.TRAIN_LABELS_FILE}")
    y_train = pd.read_csv(config.TRAIN_LABELS_FILE)['label'].values

    if args.model_type == 'nb':
        clf = GaussianNB()
    elif args.model_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}") # Should not happen
    
    logger.info(f"Training {args.model_type} model...")
    try:
        X_train_dense = X_train_features.toarray()
    except MemoryError:
        logger.error("MemoryError converting sparse features to dense. Consider alternatives.")
        raise
    
    clf.fit(X_train_dense, y_train)
    logger.info("Model training complete.")

    model_filename = f"model.joblib"
    output_path = config.MODELS_DIR / model_filename # Use Path object from config
    
    joblib.dump(clf, output_path)
    logger.info(f"Trained classifier saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a sentiment analysis classifier.')
    parser.add_argument('--model_type', choices=['nb', 'logistic'], default='nb',
                        help='Classifier type: GaussianNB or LogisticRegression.')
    
    parsed_args = parser.parse_args() # Parse arguments
    
    # --- For standalone run, ensure processed data exists ---
    if not config.TRAIN_FEATURES_FILE.exists() or not config.TRAIN_LABELS_FILE.exists():
        logger.warning(f"Processed train data not found. Please run 'python -m sentiment_src.dataset' first.")
    else:
        main(parsed_args) # Explicitly call main with the parsed arguments
