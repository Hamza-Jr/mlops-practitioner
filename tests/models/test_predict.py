from unittest.mock import Mock

import pandas as pd

from prodml.models.onnx_model import ONNXWildfireModel
from prodml.models.predictor import WildfirePredictor
from prodml.utils.config import settings


def test_prediction_output_types_and_probability_range() -> None:
    """Test that prediction returns the expected class and probability types."""

    X = pd.DataFrame(
        [
            {
                settings.feature_names[0]: 0.42,
                settings.feature_names[1]: 14500.0,
                settings.feature_names[2]: 5.0,
            }
        ]
    )

    model = ONNXWildfireModel(settings.onnx_model_path)
    predictor = WildfirePredictor(model)

    result = predictor.predict_one(X)

    assert isinstance(result["class"], str)
    assert result["class"] in settings.class_mapping

    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0


def test_prediction_is_deterministic() -> None:
    """Test that repeated predictions for the same input are identical."""

    X = pd.DataFrame(
        [
            {
                settings.feature_names[0]: 0.42,
                settings.feature_names[1]: 14500.0,
                settings.feature_names[2]: 5.0,
            }
        ]
    )

    model = ONNXWildfireModel(settings.onnx_model_path)
    predictor = WildfirePredictor(model)

    result1 = predictor.predict_one(X)
    result2 = predictor.predict_one(X)

    assert result1 == result2


def test_predict_batch_returns_class_names() -> None:
    """Test that batch predictions are converted to class names."""

    X = pd.DataFrame(
        [
            {
                settings.feature_names[0]: 0.42,
                settings.feature_names[1]: 14500.0,
                settings.feature_names[2]: 5.0,
            },
            {
                settings.feature_names[0]: 0.10,
                settings.feature_names[1]: 12000.0,
                settings.feature_names[2]: 0.0,
            },
        ]
    )

    model = Mock()
    model.predict_batch.return_value = [1, 0]

    predictor = WildfirePredictor(model)

    result = predictor.predict_batch(X)

    assert result == ["fire", "no_fire"]

    model.predict_batch.assert_called_once_with(X)
