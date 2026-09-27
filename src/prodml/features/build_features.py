from collections.abc import Mapping, Sequence

import pandas as pd


def build_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    target_column: str,
    class_mapping: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build model features and encoded target."""

    # Select model input features in the required order
    X = df[list(feature_names)].copy()

    # Encode target classes
    y = df[target_column].map(class_mapping).astype(int).copy()

    return X, y
