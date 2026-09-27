from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prodml.models.base import ModelBase


class JoblibWildfireModel(ModelBase):
    """Joblib implementation of the wildfire classification model."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        self._model = None

        self.load()

    def load(self) -> None:
        """Load the trained model artifact."""
        self._model = joblib.load(self._model_path)

    def predict_one(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class for one observation."""

        if len(X) != 1:
            raise ValueError("predict_one() expects exactly one observation.")

        return self._model.predict(X)

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes for multiple observations."""
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""
        return self._model.predict_proba(X)
