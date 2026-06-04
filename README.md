# Fake News Detector

[![GitHub](https://img.shields.io/badge/GitHub-NatiaG7-blue)](https://github.com/NatiaG7/Fake-News-Detection-in-News-Articles)

NLP pipeline and Streamlit app that classifies news article text as **real** or **fake**.

> Pre-trained models are included in `models/` — you can run the app without downloading the full dataset.

## Business problem

Media platforms and analysts need a fast first-pass filter for misleading content. This project trains a lightweight text classifier and exposes it through a simple inference UI suitable for demo and portfolio review.

## Dataset

- **Source:** Fake and Real News (ISOT-style) — `Fake.csv` + `True.csv`
- **Size:** ~45k articles combined
- **Label:** `0` = fake, `1` = real
- **Download:** See [data/README.md](data/README.md) — raw CSVs are not committed to git

## Methodology

```
Raw text → publisher-tag cleaning → TF-IDF (5k features) → Logistic Regression → label
```

1. **Preprocessing:** Remove common publisher prefixes (e.g. `WASHINGTON (Reuters) -`)
2. **Vectorization:** TF-IDF with English stop words, max 5,000 features
3. **Model:** Logistic Regression (`liblinear` solver)
4. **Evaluation:** 80/20 stratified split, accuracy + classification report

## Results

| Metric | Value |
|--------|-------|
| **Test accuracy** | **97.66%** |
| Fake class F1 | ~0.98 |
| Real class F1 | ~0.98 |

Full metrics: [outputs/metrics.json](outputs/metrics.json)

## Technologies

- Python, pandas, scikit-learn, joblib
- Streamlit (inference UI)
- Jupyter (exploratory training notebook)

## Project structure

```
fake-news-detector/
├── app.py                 # Streamlit inference app
├── src/
│   ├── data_loader.py     # Load Fake.csv + True.csv
│   ├── preprocess.py      # Text cleaning
│   ├── train.py           # Train and save artifacts
│   └── predict.py         # Inference helpers
├── models/                # Saved vectorizer + classifier
├── notebooks/             # Original training notebook
├── data/README.md         # How to download data
└── outputs/metrics.json
```

## Quick start

```bash
git clone https://github.com/NatiaG7/Fake-News-Detection-in-News-Articles.git
cd Fake-News-Detection-in-News-Articles
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the app (uses models/ artifacts)
streamlit run app.py
```

### Retrain (optional)

Place `Fake.csv` and `True.csv` in `data/raw/`, then:

```bash
python -m src.train
```

## Future improvements

- FastAPI endpoint for batch scoring
- Model comparison (Naive Bayes, linear SVM)
- Docker container for deployment
- Evaluation on user-uploaded holdout set with precision/recall dashboard

## Author

Natia Gogitidze — AI/ML Data Engineer portfolio project
