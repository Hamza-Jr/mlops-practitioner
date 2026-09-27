from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from prodml.models.base import ModelBase


class ONNXWildfireModel(ModelBase):
    """ONNX Runtime implementation of the wildfire classification model."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        self._session = None
        self.load()

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._probability_output_name = self._session.get_outputs()[1].name

    def load(self) -> None:
        """Load the ONNX wildfire model."""

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict_one(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class for one observation."""

        if len(X) != 1:
            raise ValueError("predict_one() expects exactly one observation.")

        inputs = X.to_numpy(dtype=np.float32)

        return self._session.run(
            [self._output_name],
            {self._input_name: inputs},
        )[0]

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes for multiple observations."""

        inputs = X.to_numpy(dtype=np.float32)

        return self._session.run(
            [self._output_name],
            {self._input_name: inputs},
        )[0]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""

        inputs = X.to_numpy(dtype=np.float32)

        probabilities = self._session.run(
            [self._probability_output_name],
            {self._input_name: inputs},
        )[0]

        return np.array(
            [[probability[0], probability[1]] for probability in probabilities],
            dtype=np.float32,
        )
