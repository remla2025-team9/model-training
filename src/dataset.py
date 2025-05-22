import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
from sentiment_analysis_preprocessing.preprocess import preprocess 

from . import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_generate_features(raw_data_path, text_col, label_col, sep='\t', quoting=3):
    logger.info(f"Loading raw data from: {raw_data_path}")
    dataset = pd.read_csv(raw_data_path, delimiter=sep, quoting=quoting)
    logger.info(f"Generating features from text column '{text_col}' using external 'preprocess'")
    corpus_features = preprocess(dataset[text_col].values.tolist(), save=False) 
    labels = dataset[label_col].values
    logger.info(f"Generated features of shape {corpus_features.shape} and {len(labels)} labels.")
    return corpus_features, labels

def main(args):
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_features, y_labels = load_and_generate_features(
        raw_data_path=args.raw_data_path,
        text_col=args.text_col,
        label_col=args.label_col
    )

    logger.info(f"Splitting features and labels with test_size={args.test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

    logger.info(f"Saving processed training features to {config.TRAIN_FEATURES_FILE}")
    save_npz(config.TRAIN_FEATURES_FILE, X_train)
    pd.DataFrame(y_train, columns=['label']).to_csv(config.TRAIN_LABELS_FILE, index=False)
    
    logger.info(f"Saving processed test features to {config.TEST_FEATURES_FILE}")
    save_npz(config.TEST_FEATURES_FILE, X_test)
    pd.DataFrame(y_test, columns=['label']).to_csv(config.TEST_LABELS_FILE, index=False)
    
    logger.info("Data processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load raw data, generate features, split, and save.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data for test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splitting.')
    parser.add_argument('--raw_data_path', type=str, default=str(config.DEFAULT_RAW_DATA_FILE),
                        help='Path to the raw TSV data file (overrides config).')
    parser.add_argument('--text_col', type=str, default='Review', help='Column name for text in raw data.')
    parser.add_argument('--label_col', type=str, default='Liked', help='Column name for labels in raw data.')
    
    parsed_args = parser.parse_args()
        
    main(parsed_args)