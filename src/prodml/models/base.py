from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ModelBase(ABC):
    """Interface for wildfire classification models."""

    @abstractmethod
    def load(self) -> None:
        """Load the trained model artifact."""
        ...

    @abstractmethod
    def predict_one(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class for one observation."""
        ...

    @abstractmethod
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes for multiple observations."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""
        ...
