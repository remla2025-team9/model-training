#!/usr/bin/env python
import argparse
import os
import pandas as pd
from sentiment_analysis_preprocessing.preprocess import preprocess

def main():
    p = argparse.ArgumentParser(
        description="Generate the preprocessor.joblib artifact"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic preprocessing"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to raw TSV data (e.g. data/raw/training_data.tsv)"
    )
    p.add_argument(
        "--output", required=True,
        help="Path to write the preprocessor (vectorizers/preprocessor.joblib)"
    )
    args = p.parse_args()

    # Make Python’s hash function deterministic
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load raw data
    df = pd.read_csv(args.input, delimiter="\t", quoting=3)
    # Run your library’s preprocess (which saves to the given path)
    preprocess(df["Review"].tolist(), save_path=args.output)

if __name__ == "__main__":
    main()
