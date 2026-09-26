import pandas as pd

from prodml.models.base import ModelBase
from prodml.utils.config import settings
from prodml.utils.decorators import timed


class WildfirePredictor:
    """Application interface for wildfire predictions."""

    def __init__(self, model: ModelBase) -> None:
        self._model = model
        self._class_labels = {
            class_id: class_name
            for class_name, class_id in settings.class_mapping.items()
        }

    @timed
    def predict_one(self, X: pd.DataFrame) -> dict[str, str | float]:
        """Predict one observation and return its class and confidence."""

        prediction = self._model.predict_one(X)
        probabilities = self._model.predict_proba(X)

        predicted_class = int(prediction[0])
        confidence = float(probabilities[0, predicted_class])
        class_name = self._class_labels[predicted_class]

        return {
            "class": class_name,
            "probability": confidence,
        }

    @timed
    def predict_batch(
        self,
        X: pd.DataFrame,
    ) -> list[str]:
        """Predict classes for multiple observations."""

        predictions = self._model.predict_batch(X)

        results = []

        for prediction in predictions:
            predicted_class = int(prediction)
            class_name = self._class_labels[predicted_class]
            results.append(class_name)

        return results
