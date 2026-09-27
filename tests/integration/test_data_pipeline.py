from pathlib import Path

import pandas as pd

from prodml.pipelines.data_pipeline import run_data_pipeline
from prodml.utils.config import settings


def test_run_data_pipeline_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The complete data pipeline should produce features, targets, and a processed file."""

    raw_data = pd.DataFrame(
        {
            "NDVI": [0.10, 0.20, 0.30],
            "LST": [300.0, 305.0, 310.0],
            "BURNED_AREA": [0.0, 5.0, 8.0],
            "CLASS": ["no_fire", "fire", "fire"],
        }
    )

    input_path = tmp_path / "raw.csv"
    raw_data.to_csv(input_path, sep=";", index=False)

    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        settings,
        "processed_data_dir",
        processed_dir,
    )

    X, y = run_data_pipeline(input_path=input_path)

    # Verify returned features.
    assert list(X.columns) == [
        "NDVI",
        "LST",
        "BURNED_AREA",
    ]

    assert len(X) == 3

    # Verify encoded target.
    assert y.tolist() == [0, 1, 1]

    # Verify processed artifact was written.
    processed_path = processed_dir / settings.processed_dataset_filename

    assert processed_path.exists()

    # Verify the saved dataset can be loaded.
    processed_df = pd.read_csv(processed_path)

    assert list(processed_df.columns) == [
        "NDVI",
        "LST",
        "BURNED_AREA",
        "CLASS",
    ]

    assert len(processed_df) == 3
