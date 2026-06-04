"""Load trained artifacts and run inference."""

from pathlib import Path

import joblib

from src.preprocess import clean_text_publisher_tag

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_artifacts(model_dir: Path | None = None):
    """Load TF-IDF vectorizer and logistic regression model."""
    base = model_dir or MODEL_DIR
    vectorizer = joblib.load(base / "tfidf_vectorizer.pkl")
    model = joblib.load(base / "fake_news_model.pkl")
    return vectorizer, model


def predict_text(text: str, model_dir: Path | None = None) -> dict:
    """
    Classify article text as fake (0) or real (1).

    Returns dict with label, label_name, and raw prediction.
    """
    vectorizer, model = load_artifacts(model_dir)
    cleaned = clean_text_publisher_tag(text)
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    return {
        "label": prediction,
        "label_name": "real" if prediction == 1 else "fake",
        "is_real": prediction == 1,
    }
