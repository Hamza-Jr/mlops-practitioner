from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def validate_data(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    target_column: str,
    class_mapping: Mapping[str, int],
) -> None:
    """Validate that input data satisfies the expected schema and quality rules."""

    # 1. Check that the dataset is not empty.
    if df.empty:
        raise ValueError("Input dataset is empty.")

    # 2. Check required feature and target columns.
    required_columns = [*feature_names, target_column]

    missing_columns = [name for name in required_columns if name not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 3. Extract feature columns.
    features = df[list(feature_names)]

    # 4. Check feature data types.
    non_numeric = [
        column
        for column in feature_names
        if not pd.api.types.is_numeric_dtype(features[column])
    ]

    if non_numeric:
        raise TypeError(f"Non-numeric feature columns detected: {non_numeric}")

    # 5. Check infinite feature values.
    if np.isinf(features.to_numpy(dtype=np.float64)).any():
        raise ValueError("Feature columns contain infinite values (inf or -inf).")

    # 6. Check target labels.
    valid_labels = set(class_mapping)

    invalid_labels = set(df[target_column].dropna()) - valid_labels

    if invalid_labels:
        raise ValueError(f"Unknown target labels detected: {sorted(invalid_labels)}")
