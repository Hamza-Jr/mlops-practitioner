from pathlib import Path

import pandas as pd

from prodml.data.ingestion import load_data
from prodml.data.preprocessing import preprocess_data
from prodml.data.validation import validate_data
from prodml.features.build_features import build_features
from prodml.utils.config import settings


def run_data_pipeline(
    input_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the complete data preparation pipeline."""

    # ===============================================================
    # 1. Load raw data
    # ===============================================================

    data_path = input_path or settings.dataset_path

    df = load_data(data_path)

    # ===============================================================
    # 2. Validate raw data
    # ===============================================================

    validate_data(
        df=df,
        feature_names=settings.feature_names,
        target_column=settings.target_column,
        class_mapping=settings.class_mapping,
    )

    # ===============================================================
    # 3. Preprocess data
    # ===============================================================

    df = preprocess_data(
        df=df,
        feature_names=settings.feature_names,
    )

    # ===============================================================
    # 4. Save processed data
    # ===============================================================

    settings.processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_path = settings.processed_data_dir / settings.processed_dataset_filename

    df.to_csv(
        processed_path,
        index=False,
    )

    # ===============================================================
    # 5. Build features
    # ===============================================================

    X, y = build_features(
        df=df,
        feature_names=settings.feature_names,
        target_column=settings.target_column,
        class_mapping=settings.class_mapping,
    )

    return X, y
