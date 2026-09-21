from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """Train a classification model."""

    model.fit(X_train, y_train)

    return model


def save_model(
    model: BaseEstimator,
    model_path: Path,
) -> None:
    """Save a trained model to a joblib artifact."""

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )
