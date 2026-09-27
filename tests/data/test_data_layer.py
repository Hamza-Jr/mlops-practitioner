from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prodml.data.ingestion import load_data
from prodml.data.preprocessing import preprocess_data
from prodml.data.validation import validate_data

FEATURES = ("NDVI", "LST", "BURNED_AREA")
TARGET = "CLASS"
CLASS_MAPPING = {
    "no_fire": 0,
    "fire": 1,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Return a small valid wildfire dataset."""
    return pd.DataFrame(
        {
            "NDVI": [0.10, 0.20, 0.30],
            "LST": [300.0, 305.0, 310.0],
            "BURNED_AREA": [0.0, 5.0, 8.0],
            "CLASS": ["no_fire", "fire", "fire"],
        }
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_load_data_reads_csv(tmp_path: Path, valid_dataframe: pd.DataFrame) -> None:
    """load_data should correctly load a valid CSV file."""
    csv_path = tmp_path / "wildfire.csv"
    valid_dataframe.to_csv(csv_path, sep=";", index=False)

    result = load_data(csv_path)

    pd.testing.assert_frame_equal(result, valid_dataframe)


def test_load_data_raises_when_file_does_not_exist(tmp_path: Path) -> None:
    """load_data should raise FileNotFoundError for a missing file."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(missing_path)


def test_load_data_raises_when_path_is_directory(tmp_path: Path) -> None:
    """load_data should reject a directory path."""
    with pytest.raises(ValueError, match="not a file"):
        load_data(tmp_path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_data_converts_numeric_features() -> None:
    """preprocess_data should convert numeric strings to numbers."""
    df = pd.DataFrame(
        {
            "NDVI": ["0.10", "0.20", "0.30"],
            "LST": ["300", "305", "310"],
            "BURNED_AREA": ["0", "5", "8"],
            "CLASS": ["no_fire", "fire", "fire"],
        }
    )

    result = preprocess_data(df, FEATURES)

    for feature in FEATURES:
        assert pd.api.types.is_numeric_dtype(result[feature])


def test_preprocess_data_interpolates_missing_values() -> None:
    """preprocess_data should interpolate missing values between observations."""
    df = pd.DataFrame(
        {
            "NDVI": [0.10, np.nan, 0.30],
            "LST": [300.0, 305.0, 310.0],
            "BURNED_AREA": [0.0, 5.0, 8.0],
            "CLASS": ["no_fire", "fire", "fire"],
        }
    )

    result = preprocess_data(df, FEATURES)

    assert result["NDVI"].tolist() == [0.10, 0.20, 0.30]


def test_preprocess_data_drops_unfillable_missing_values() -> None:
    """Missing values at the boundaries should be removed."""
    df = pd.DataFrame(
        {
            "NDVI": [np.nan, 0.20, 0.30],
            "LST": [300.0, 305.0, 310.0],
            "BURNED_AREA": [0.0, 5.0, 8.0],
            "CLASS": ["no_fire", "fire", "fire"],
        }
    )

    result = preprocess_data(df, FEATURES)

    assert len(result) == 2
    assert result["NDVI"].isna().sum() == 0


def test_preprocess_data_removes_duplicates() -> None:
    """preprocess_data should remove exact duplicate rows."""
    df = pd.DataFrame(
        {
            "NDVI": [0.10, 0.10, 0.20],
            "LST": [300.0, 300.0, 305.0],
            "BURNED_AREA": [0.0, 0.0, 5.0],
            "CLASS": ["no_fire", "no_fire", "fire"],
        }
    )

    result = preprocess_data(df, FEATURES)

    assert len(result) == 2


def test_preprocess_data_does_not_modify_original_dataframe(
    valid_dataframe: pd.DataFrame,
) -> None:
    """preprocess_data should not mutate the caller's DataFrame."""
    original = valid_dataframe.copy()

    preprocess_data(valid_dataframe, FEATURES)

    pd.testing.assert_frame_equal(valid_dataframe, original)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_data_accepts_valid_dataframe(
    valid_dataframe: pd.DataFrame,
) -> None:
    """validate_data should accept a valid dataset."""
    validate_data(
        df=valid_dataframe,
        feature_names=FEATURES,
        target_column=TARGET,
        class_mapping=CLASS_MAPPING,
    )


def test_validate_data_rejects_empty_dataframe() -> None:
    """validate_data should reject an empty dataset."""
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input dataset is empty"):
        validate_data(
            df=df,
            feature_names=FEATURES,
            target_column=TARGET,
            class_mapping=CLASS_MAPPING,
        )


def test_validate_data_rejects_missing_columns() -> None:
    """validate_data should reject datasets missing required columns."""
    df = pd.DataFrame(
        {
            "NDVI": [0.1],
            "LST": [300.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(
            df=df,
            feature_names=FEATURES,
            target_column=TARGET,
            class_mapping=CLASS_MAPPING,
        )


def test_validate_data_rejects_non_numeric_features() -> None:
    """validate_data should reject non-numeric feature columns."""
    df = pd.DataFrame(
        {
            "NDVI": ["invalid"],
            "LST": [300.0],
            "BURNED_AREA": [5.0],
            "CLASS": ["fire"],
        }
    )

    with pytest.raises(TypeError, match="Non-numeric feature columns"):
        validate_data(
            df=df,
            feature_names=FEATURES,
            target_column=TARGET,
            class_mapping=CLASS_MAPPING,
        )


def test_validate_data_rejects_infinite_features() -> None:
    """validate_data should reject inf and -inf feature values."""
    df = pd.DataFrame(
        {
            "NDVI": [np.inf],
            "LST": [300.0],
            "BURNED_AREA": [5.0],
            "CLASS": ["fire"],
        }
    )

    with pytest.raises(ValueError, match="infinite values"):
        validate_data(
            df=df,
            feature_names=FEATURES,
            target_column=TARGET,
            class_mapping=CLASS_MAPPING,
        )


def test_validate_data_rejects_unknown_target_label() -> None:
    """validate_data should reject target labels outside the class mapping."""
    df = pd.DataFrame(
        {
            "NDVI": [0.1],
            "LST": [300.0],
            "BURNED_AREA": [5.0],
            "CLASS": ["unknown"],
        }
    )

    with pytest.raises(ValueError, match="Unknown target labels"):
        validate_data(
            df=df,
            feature_names=FEATURES,
            target_column=TARGET,
            class_mapping=CLASS_MAPPING,
        )
