# Data

Training data is **not** committed to this repository (file size + licensing).

## Dataset

Use the **Fake and Real News** dataset (ISOT-style layout):

- `Fake.csv` — fake articles
- `True.csv` — real articles

Common sources:

- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Or the original ISOT publication dataset

## Setup

1. Download the dataset.
2. Place files here:

```
data/raw/Fake.csv
data/raw/True.csv
```

3. Retrain (optional — pre-trained artifacts are in `models/`):

```bash
python -m src.train
```

## Columns used

| Column | Usage |
|--------|--------|
| `text` | Model input after light cleaning |
| `title`, `subject`, `date` | Loaded but not used in baseline model |
