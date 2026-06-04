"""Train TF-IDF + Logistic Regression fake news classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocess import clean_text_publisher_tag

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def train(
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Train model, save artifacts and metrics."""
    data_dir = data_dir or DATA_RAW
    model_dir = model_dir or MODEL_DIR
    output_dir = output_dir or OUTPUT_DIR

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_dir)
    df["text"] = df["text"].apply(clean_text_publisher_tag)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(random_state=42, solver="liblinear")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(model, model_dir / "fake_news_model.pkl")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")

    metrics = {
        "accuracy": round(accuracy, 4),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "classification_report": report,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    return metrics


if __name__ == "__main__":
    train()
