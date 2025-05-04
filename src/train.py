import argparse
import logging
import os
import joblib
import pandas as pd
from sentiment_analysis_preprocessing.preprocess import preprocess
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import TransformerMixin

def load_data(path, text_col, label_col, sep='\t', quoting=3):
    dataset = pd.read_csv(path, delimiter=sep, quoting=quoting)

    # Preprocess text and return as sparse matrix
    corpus = preprocess(dataset['Review'].values.tolist())

    # Separate the labels
    labels = dataset.iloc[:, -1].values

    return corpus, labels

def main():
    parser = argparse.ArgumentParser(description='Train & save sentiment analysis pipeline')
    parser.add_argument('--data-path', type=str, default='data/training_data.tsv',
                        help='Path to TSV with text and labels')
    parser.add_argument('--text-col', type=str, default='Review',
                        help='Column name for text')
    parser.add_argument('--label-col', type=str, default='Liked',
                        help='Column name for sentiment labels')
    parser.add_argument('--vectorizer', choices=['count', 'tfidf'], default='count',
                        help='Type of vectorizer')
    parser.add_argument('--max-features', type=int, default=1420,
                        help='Max features for vectorizer')
    parser.add_argument('--model-type', choices=['nb', 'logistic'], default='nb',
                        help='Classifier: GaussianNB or LogisticRegression')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for test split')
    parser.add_argument("--version", type=str, default='latest',
                        help="Model version to embed in the output filename")
    parser.add_argument('--output-path', type=str, default='models/model.joblib',
                        help='Where to save the pipeline')
    args = parser.parse_args()

    if not args.version:
        raise ValueError("You must provide the version with --version <version>")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load and preprocess
    logger.info('Loading data from %s', args.data_path)
    X, y = load_data(args.data_path, args.text_col, args.label_col)


    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)
    logger.info('Train samples: %d, Test samples: %d', X_train.shape[0],X_test.shape[0])

    
    if args.model_type == 'nb':
        clf = GaussianNB()
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)

    # Train
    logger.info('Training model...')
    clf.fit(X_train.toarray(), y_train)

    # Evaluate
    logger.info('Evaluating on test split...')
    y_pred = clf.predict(X_test.toarray())
    logger.info('Classification report:\n%s', classification_report(y_test, y_pred))
    logger.info('Confusion matrix:\n%s', confusion_matrix(y_test, y_pred))

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    base, ext = os.path.splitext(args.output_path)
    
    if args.version == 'latest':
        versioned_path = f"{base}-{args.version}{ext or '.joblib'}"
    else:
        versioned_path = f"{base}-v{args.version}{ext or '.joblib'}"
    
    joblib.dump(clf, versioned_path)
    logger.info('Pipeline saved to %s', versioned_path)

if __name__ == '__main__':
    main()