import tempfile
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prodml.api.dependencies import get_predictor
from prodml.api.main import app
from prodml.models.export import export_model
from prodml.models.factory import create_model
from prodml.models.onnx_model import ONNXWildfireModel
from prodml.models.predictor import WildfirePredictor
from prodml.models.train import save_model, train_model
from prodml.utils.config import settings


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Provide valid wildfire features for prediction tests."""

    return pd.DataFrame(
        {
            "NDVI": [0.45, 0.12, 0.78, 0.33],
            "LST": [14500.0, 15000.0, 14000.0, 14800.0],
            "BURNED_AREA": [3.0, 5.0, 9.0, 4.0],
        }
    )[list(settings.feature_names)]


@pytest.fixture(scope="session")
def trained_model() -> Generator[ONNXWildfireModel, None, None]:
    """Train and export one small ONNX model for the test session."""

    X_train = pd.DataFrame(
        {
            "NDVI": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
            "LST": [14000.0, 14500.0, 15000.0, 15500.0, 14200.0, 15200.0],
            "BURNED_AREA": [3.0, 5.0, 8.0, 9.0, 4.0, 7.0],
        }
    )[list(settings.feature_names)]

    y_train = pd.Series(
        [1, 1, 0, 0, 1, 0],
        name=settings.target_column,
    )

    model = create_model("random_forest")

    trained = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        model_path = temp_path / "test_model.pkl"
        onnx_path = temp_path / "test_model.onnx"

        save_model(
            model=trained,
            model_path=model_path,
        )

        export_model(
            input_path=model_path,
            output_path=onnx_path,
        )

        yield ONNXWildfireModel(onnx_path)


@pytest.fixture
def client(
    trained_model: ONNXWildfireModel,
) -> Generator[TestClient, None, None]:
    """Provide a FastAPI client using the test-trained ONNX model."""

    predictor = WildfirePredictor(trained_model)

    app.dependency_overrides[get_predictor] = lambda: predictor

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
