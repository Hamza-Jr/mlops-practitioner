import pandas as pd
import pytest

from prodml.features.build_features import build_features

FEATURES = ("NDVI", "LST", "BURNED_AREA")

CLASS_MAPPING = {
    "no_fire": 0,
    "fire": 1,
}


@pytest.fixture
def wildfire_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "NDVI": [0.10, 0.20, 0.30],
            "LST": [300.0, 305.0, 310.0],
            "BURNED_AREA": [0.0, 5.0, 8.0],
            "CLASS": ["no_fire", "fire", "fire"],
            "EXTRA_COLUMN": ["a", "b", "c"],
        }
    )


def test_build_features_selects_required_features(
    wildfire_dataframe: pd.DataFrame,
) -> None:
    """Only model features should be included in X."""
    X, _ = build_features(
        df=wildfire_dataframe,
        feature_names=FEATURES,
        target_column="CLASS",
        class_mapping=CLASS_MAPPING,
    )

    assert list(X.columns) == list(FEATURES)


def test_build_features_preserves_feature_order(
    wildfire_dataframe: pd.DataFrame,
) -> None:
    """Features must remain in the exact model contract order."""
    X, _ = build_features(
        df=wildfire_dataframe,
        feature_names=("LST", "NDVI", "BURNED_AREA"),
        target_column="CLASS",
        class_mapping=CLASS_MAPPING,
    )

    assert list(X.columns) == [
        "LST",
        "NDVI",
        "BURNED_AREA",
    ]


def test_build_features_encodes_target(
    wildfire_dataframe: pd.DataFrame,
) -> None:
    """Target labels should be converted using the class mapping."""
    _, y = build_features(
        df=wildfire_dataframe,
        feature_names=FEATURES,
        target_column="CLASS",
        class_mapping=CLASS_MAPPING,
    )

    assert y.tolist() == [0, 1, 1]
    assert pd.api.types.is_integer_dtype(y)


def test_build_features_does_not_modify_original_dataframe(
    wildfire_dataframe: pd.DataFrame,
) -> None:
    """Building features must not mutate the input DataFrame."""
    original = wildfire_dataframe.copy()

    build_features(
        df=wildfire_dataframe,
        feature_names=FEATURES,
        target_column="CLASS",
        class_mapping=CLASS_MAPPING,
    )

    pd.testing.assert_frame_equal(wildfire_dataframe, original)


def test_build_features_unknown_class_becomes_invalid_target() -> None:
    """An unmapped target label should not silently become a valid class."""
    df = pd.DataFrame(
        {
            "NDVI": [0.10],
            "LST": [300.0],
            "BURNED_AREA": [5.0],
            "CLASS": ["unknown"],
        }
    )

    with pytest.raises((ValueError, TypeError)):
        build_features(
            df=df,
            feature_names=FEATURES,
            target_column="CLASS",
            class_mapping=CLASS_MAPPING,
        )
