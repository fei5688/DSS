"""
Data processing for Titanic dataset.
"""

import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
LABEL = "Survived"


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def save_data(data_path: str, df: pd.DataFrame) -> None:
    df.to_csv(data_path.replace(".csv", "_processed.csv"), index=False)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 只保留训练需要的列
    required_cols = FEATURES + [LABEL]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()

    # 统一字符串格式
    df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()
    df["Embarked"] = df["Embarked"].astype(str).str.strip().str.upper()

    return df


def run(data_path: str) -> pd.DataFrame:
    logging.info("Load data...")
    df = load_data(data_path)

    logging.info("Processing data...")
    df = preprocess(df)

    logging.info("Save processed data...")
    save_data(data_path, df)

    logging.info("Completed")
    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, required=True)
    args = argparser.parse_args()
    run(args.data_path)
