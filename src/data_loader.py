"""Load and combine fake/true news CSV datasets."""

from pathlib import Path

import pandas as pd


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv from data_dir, add label column, shuffle.

    label: 0 = fake, 1 = real
    """
    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Expected Fake.csv and True.csv in {data_dir}. "
            "See data/README.md for download instructions."
        )

    fake_news = pd.read_csv(fake_path)
    true_news = pd.read_csv(true_path)

    fake_news["label"] = 0
    true_news["label"] = 1

    combined = pd.concat([fake_news, true_news], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)
