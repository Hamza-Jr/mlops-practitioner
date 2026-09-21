from collections.abc import Sequence

import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Clean the raw wildfire dataset."""

    df = df.copy()

    # 1. Convert feature columns to numeric.
    #    Invalid values become NaN and are handled below.
    df[list(feature_names)] = df[list(feature_names)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # 2. Interpolate missing numerical features.
    df[list(feature_names)] = df[list(feature_names)].interpolate(method="linear")

    # 3. Remove rows where interpolation could not fill
    #    missing values, such as NaNs at the beginning or end.
    df = df.dropna(subset=list(feature_names))

    # 4. Remove exact duplicate rows.
    df = df.drop_duplicates()

    return df
