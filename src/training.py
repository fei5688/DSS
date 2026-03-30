"""
Training script for Titanic survival prediction.
"""

import argparse
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src import data_processor
from src import model_registry

logging.basicConfig(level=logging.INFO)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
NUMERICAL_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
LABEL = "Survived"


def run(data_path: str):
    logging.info("Process Data...")
    df = data_processor.run(data_path)

    X = df[FEATURES]
    y = df[LABEL]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info("Start Training...")
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=42
            ))
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    mdl_meta = {
        "name": "titanic_model",
        "metrics": {
            "accuracy": float(acc)
        }
    }

    logging.info("Persisting model...")
    model_registry.register(clf, FEATURES, mdl_meta)

    logging.info("Training completed.")
    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, required=True)
    args = argparser.parse_args()
    run(args.data_path)
