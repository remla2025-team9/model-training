import argparse
import os
import pandas as pd
import shutil
from sentiment_analysis_preprocessing.preprocess import preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    df = pd.read_csv(args.input, delimiter="\t", quoting=3)
    model_path = args.output
    data_path = "vectorizers/preprocessed_data.joblib"

    preprocess(
        df["Review"].tolist(),
        save=True,
        model_path=model_path,
        data_path=data_path
    )

    src = os.path.join("output", model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    shutil.move(src, model_path)


if __name__ == "__main__":
    main()
