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

    # Ensure deterministic hashing
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load raw data
    df = pd.read_csv(args.input, delimiter="\t", quoting=3)

    # Library preprocess API: preprocess(data, save=True, model_path, data_path)
    preprocess(
        df["Review"].tolist(),
        save=True,
        model_path=args.output,
        data_path="vectorizers/preprocessed_data.joblib"
    )


if __name__ == "__main__":
    main()
