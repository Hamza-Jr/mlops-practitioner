import numpy as np
import pandas as pd

from prodml.models.onnx_model import ONNXWildfireModel
from prodml.models.pickle_model import JoblibWildfireModel
from prodml.utils.config import settings


def test_pickle_onnx_parity() -> None:
    """Verify that Pickle and ONNX models produce equivalent outputs."""

    # Load the fixed 500-row validation dataset.
    validation_data = pd.read_csv(settings.validation_data_path)

    assert len(validation_data) == 500

    X = validation_data[list(settings.feature_names)]

    # Load both models.
    pickle_model = JoblibWildfireModel(settings.model_path)

    onnx_model = ONNXWildfireModel(
        settings.artifacts_dir / f"{settings.model_name}.onnx"
    )

    # Predict classes with both models.
    pred_pkl = pickle_model.predict_batch(X)
    pred_onnx = onnx_model.predict_batch(X)

    # Compare class predictions.
    predictions_match = np.allclose(
        pred_pkl,
        pred_onnx,
        atol=1e-4,
    )

    # Predict probabilities with both models.
    proba_pkl = pickle_model.predict_proba(X)
    proba_onnx = onnx_model.predict_proba(X)

    # Compare probabilities.
    probabilities_match = np.allclose(
        proba_pkl,
        proba_onnx,
        atol=1e-4,
    )

    assert predictions_match
    assert probabilities_match
