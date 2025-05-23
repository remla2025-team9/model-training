import argparse
import logging
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

from sentiment_analysis_preprocessing.preprocess import preprocess
from . import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def move_output_to_vectorizers(output_dir='output', target_dir='vectorizers'):
    """
    Move files from the default output directory of the preprocessing library ('output')
    to the 'vectorizers' directory for better organization.
    """
    if not os.path.exists(output_dir):
        logger.warning(f"Output directory '{output_dir}' does not exist, skipping move.")
        return
    
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Move each file from output_dir to target_dir
    for filename in os.listdir(output_dir):
        src_path = os.path.join(output_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        logger.info(f"Moving '{src_path}' to '{dst_path}'")
        shutil.move(src_path, dst_path)
    
    # Attempt to remove the now empty output directory
    try:
        os.rmdir(output_dir)
        logger.info(f"Removed empty output directory '{output_dir}'.")
    except OSError:
        logger.warning(f"Could not remove output directory '{output_dir}', it may not be empty.")

def load_and_generate_features(raw_data_path, text_col, label_col, sep='\t', quoting=3):
    """
    Load raw data from a CSV/TSV file, generate features by preprocessing the text column,
    and return the feature matrix and label array.
    
    The preprocess function saves vectorizer model and preprocessed data files 
    to a default 'output' directory, which we later move to 'vectorizers'.
    """
    logger.info(f"Loading raw data from: {raw_data_path}")
    dataset = pd.read_csv(raw_data_path, delimiter=sep, quoting=quoting)
    
    logger.info(f"Generating features from text column '{text_col}' using external 'preprocess'")

    # Use simple filenames for model and data; preprocess saves them to './output/' by default
    model_path = 'preprocessor.joblib'
    data_path = 'preprocessed_data.joblib'

    corpus_features = preprocess(
        dataset[text_col].values.tolist(),
        save=True,
        model_path=model_path,
        data_path=data_path
    )

    # Move the files generated in 'output' directory to 'vectorizers' directory
    move_output_to_vectorizers(output_dir='output', target_dir='vectorizers')

    labels = dataset[label_col].values
    logger.info(f"Generated features of shape {corpus_features.shape} and {len(labels)} labels.")
    return corpus_features, labels

def main(args):
    # Ensure the directory for processed data exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data, generate features and labels
    X_features, y_labels = load_and_generate_features(
        raw_data_path=args.raw_data_path,
        text_col=args.text_col,
        label_col=args.label_col
    )

    logger.info(f"Splitting features and labels with test_size={args.test_size}")
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

    # Save the training features and labels to disk
    logger.info(f"Saving processed training features to {config.TRAIN_FEATURES_FILE}")
    save_npz(config.TRAIN_FEATURES_FILE, X_train)
    pd.DataFrame(y_train, columns=['label']).to_csv(config.TRAIN_LABELS_FILE, index=False)
    
    # Save the test features and labels to disk
    logger.info(f"Saving processed test features to {config.TEST_FEATURES_FILE}")
    save_npz(config.TEST_FEATURES_FILE, X_test)
    pd.DataFrame(y_test, columns=['label']).to_csv(config.TEST_LABELS_FILE, index=False)
    
    logger.info("Data processing complete.")

if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Load raw data, generate features, split, and save.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data for test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splitting.')
    parser.add_argument('--raw_data_path', type=str, default=str(config.DEFAULT_RAW_DATA_FILE),
                        help='Path to the raw TSV data file (overrides config).')
    parser.add_argument('--text_col', type=str, default='Review', help='Column name for text in raw data.')
    parser.add_argument('--label_col', type=str, default='Liked', help='Column name for labels in raw data.')
    
    args = parser.parse_args()
    main(args)
