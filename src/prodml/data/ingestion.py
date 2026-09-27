from pathlib import Path

import pandas as pd


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the raw wildfire dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Dataset path is not a file: {file_path}")

    return pd.read_csv(file_path, sep=";")
