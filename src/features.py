"""Feature generation and data preprocessing routines."""

import argparse
import logging
import dvc.api
import pandas as pd
import joblib
from sentiment_analysis_preprocessing.preprocesser import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

from . import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_features(dataset, text_col, label_col):
    """
    Generate features from text data using Preprocessor and CountVectorizer.

    Args:
        dataset: pandas DataFrame with raw data
        text_col: Name of the text column
        label_col: Name of the label column

    Returns:
        tuple: (features_matrix, labels_array)
    """
    logger.info(f"Preprocessing text from column '{text_col}' using external Preprocessor")

    # Use the Preprocessor class to preprocess the text data
    preprocessor = Preprocessor()
    preprocessed_texts = preprocessor.transform(dataset[text_col].values.tolist())

    # Convert to list if needed
    if not isinstance(preprocessed_texts, list):
        preprocessed_texts = preprocessed_texts.tolist()

    logger.info("Generating Count features from preprocessed text")

    # Create and fit Count vectorizer on preprocessed text
    vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Fit and transform the preprocessed text data
    corpus_features = vectorizer.fit_transform(preprocessed_texts)

    # Save the fitted vectorizer
    config.VECTORIZERS_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.VECTORIZERS_DIR / "vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Saved vectorizer to {vectorizer_path}")

    labels = dataset[label_col].values
    logger.info(f"Generated sparse features with shape {corpus_features.shape} and {len(labels)} labels.")

    return corpus_features, labels


def split_and_save_data(features, labels, test_size=0.2, random_state=42):
    """
    Split features and labels into train/test sets and save to disk.

    Args:
        features: Feature matrix
        labels: Label array
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
    """
    logger.info(f"Splitting features and labels with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

    # Ensure the directory for processed data exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the training features and labels to disk
    logger.info(f"Saving processed training features to {config.TRAIN_FEATURES_FILE}")
    save_npz(config.TRAIN_FEATURES_FILE, X_train)
    pd.DataFrame(y_train, columns=["label"]).to_csv(config.TRAIN_LABELS_FILE, index=False)

    # Save the test features and labels to disk
    logger.info(f"Saving processed test features to {config.TEST_FEATURES_FILE}")
    save_npz(config.TEST_FEATURES_FILE, X_test)
    pd.DataFrame(y_test, columns=["label"]).to_csv(config.TEST_LABELS_FILE, index=False)

    logger.info("Data processing complete.")


def main(args, test_size=0.2):
    """
    Main function to load data, generate features, and save processed data.
    """
    # Set raw data path
    raw_data_path = config.DEFAULT_RAW_DATA_FILE
    logger.info(f"Loading raw TSV data from: {raw_data_path}")

    # Read TSV file using pandas.read_csv with tab delimiter
    dataset = pd.read_csv(raw_data_path, delimiter='\t', quoting=3)
    logger.info(f"Loaded dataset with shape: {dataset.shape}")

    # Generate features and labels
    X_features, y_labels = generate_features(
        dataset=dataset,
        text_col=args.text_col,
        label_col=args.label_col
    )

    # Split and save processed data
    split_and_save_data(
        features=X_features,
        labels=y_labels,
        test_size=test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    params = dvc.api.params_show()

    # Define command line arguments
    parser = argparse.ArgumentParser(
        description="Generate features from raw data, split, and save."
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for splitting."
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default=str(config.DEFAULT_RAW_DATA_FILE),
        help="Path to the raw TSV data file (overrides config).",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="Review",
        help="Column name for text in raw data.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="Liked",
        help="Column name for labels in raw data.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args, test_size=params["prepare"]["test_size"])
