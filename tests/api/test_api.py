from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from prodml.api.dependencies import get_predictor
from prodml.api.main import app


@pytest.fixture
def mock_predictor() -> Mock:
    """Create a mock predictor for API tests."""
    predictor = Mock()

    predictor.predict_one.return_value = {
        "class": "fire",
        "probability": 0.75,
    }

    predictor.predict_batch.return_value = [
        "fire",
        "no_fire",
    ]

    return predictor


@pytest.fixture
def client(mock_predictor: Mock):
    """Provide a TestClient with the predictor dependency overridden."""
    app.dependency_overrides[get_predictor] = lambda: mock_predictor

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health_returns_200(client: TestClient) -> None:
    """GET /health returns a healthy response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_prediction(
    client: TestClient,
    mock_predictor: Mock,
) -> None:
    """POST /predict returns the predictor result through the API contract."""
    payload = {
        "ndvi": 0.42,
        "lst": 14500.0,
        "burned_area": 5.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "class_name": "fire",
        "probability": 0.75,
    }

    mock_predictor.predict_one.assert_called_once()


def test_predict_response_schema(
    client: TestClient,
    mock_predictor: Mock,
) -> None:
    """POST /predict returns the expected response schema."""
    mock_predictor.predict_one.return_value = {
        "class": "no_fire",
        "probability": 0.25,
    }

    payload = {
        "ndvi": 0.42,
        "lst": 14500.0,
        "burned_area": 5.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert set(data) == {"class_name", "probability"}
    assert data["class_name"] in {"fire", "no_fire"}
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0


@pytest.mark.parametrize(
    "invalid_payload",
    [
        # Missing required field.
        {
            "lst": 14500.0,
            "burned_area": 5.0,
        },
        # Invalid type.
        {
            "ndvi": "invalid_string",
            "lst": 14500.0,
            "burned_area": 5.0,
        },
        # Value below allowed minimum.
        {
            "ndvi": 0.42,
            "lst": -10.0,
            "burned_area": 5.0,
        },
        # Value below allowed minimum.
        {
            "ndvi": 0.42,
            "lst": 14500.0,
            "burned_area": -1.0,
        },
        # Unknown field.
        {
            "ndvi": 0.42,
            "lst": 14500.0,
            "burned_area": 5.0,
            "unexpected_feature": 99.0,
        },
    ],
)
def test_predict_invalid_payload_returns_422(
    client: TestClient,
    invalid_payload: dict,
) -> None:
    """POST /predict rejects invalid request payloads."""
    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_batch_returns_predictions(
    client: TestClient,
    mock_predictor: Mock,
) -> None:
    """POST /predict/batch returns predictions for multiple observations."""
    payload = {
        "instances": [
            {
                "ndvi": 0.42,
                "lst": 14500.0,
                "burned_area": 5.0,
            },
            {
                "ndvi": 0.10,
                "lst": 12000.0,
                "burned_area": 0.0,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "predictions": ["fire", "no_fire"],
    }

    mock_predictor.predict_batch.assert_called_once()


def test_predict_batch_invalid_payload_returns_422(
    client: TestClient,
) -> None:
    """POST /predict/batch rejects an empty batch."""
    payload = {
        "instances": [],
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()
