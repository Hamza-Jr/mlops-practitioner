import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from prodml.api.dependencies import get_predictor
from prodml.api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.models.predictor import WildfirePredictor
from prodml.utils.config import settings
from prodml.utils.decorators import prediction_latency_ms

router = APIRouter()

logger = logging.getLogger("prodml.api")


def _log_outside_training_range(
    *,
    request: Request,
    features: dict[str, float],
    batch_size: int | None = None,
) -> None:
    """Log a warning when an input is outside the observed training range."""

    outside_features = []

    for feature_name, value in features.items():
        minimum, maximum = settings.training_ranges[feature_name]

        if value < minimum or value > maximum:
            outside_features.append(feature_name)

    if outside_features:
        extra = {
            "event": "input_outside_training_range",
            "endpoint": request.url.path,
            "method": request.method,
        }

        if batch_size is not None:
            extra["batch_size"] = batch_size

        logger.warning(
            "Input outside observed training range",
            extra=extra,
        )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    payload: PredictionRequest,
    predictor: WildfirePredictor = Depends(get_predictor),  # noqa: B008
) -> PredictionResponse:
    """Return a prediction for a single wildfire observation."""

    logger.info(
        "Prediction requested",
        extra={
            "event": "prediction_requested",
            "model_version": settings.model_version,
            "endpoint": "/predict",
            "method": request.method,
        },
    )

    features = {
        "ndvi": payload.ndvi,
        "lst": payload.lst,
        "burned_area": payload.burned_area,
    }

    _log_outside_training_range(
        request=request,
        features=features,
    )

    X = pd.DataFrame([features])

    logger.debug(
        "Feature vector prepared for prediction",
        extra={
            "event": "feature_vector",
            "endpoint": "/predict",
            "features": features,
        },
    )

    try:
        result = predictor.predict_one(X)

    except Exception as exc:
        logger.error(
            "Prediction failed",
            extra={
                "event": "prediction_failed",
                "model_version": settings.model_version,
                "endpoint": "/predict",
                "method": request.method,
                "status_code": 500,
                "error_type": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    logger.info(
        "Prediction served successfully",
        extra={
            "event": "prediction_successful",
            "model_version": settings.model_version,
            "endpoint": "/predict",
            "method": request.method,
            "status_code": 200,
            "latency_ms": prediction_latency_ms.get(),
        },
    )

    return PredictionResponse(
        class_name=result["class"],
        probability=float(result["probability"]),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
    predictor: WildfirePredictor = Depends(get_predictor),  # noqa: B008
) -> BatchPredictionResponse:
    """Return predictions for a batch of wildfire observations."""

    batch_size = len(payload.instances)

    logger.info(
        "Prediction requested",
        extra={
            "event": "prediction_requested",
            "model_version": settings.model_version,
            "endpoint": "/predict/batch",
            "method": request.method,
            "batch_size": batch_size,
        },
    )

    feature_records = [
        {
            "ndvi": item.ndvi,
            "lst": item.lst,
            "burned_area": item.burned_area,
        }
        for item in payload.instances
    ]

    for features in feature_records:
        _log_outside_training_range(
            request=request,
            features=features,
            batch_size=batch_size,
        )

    X = pd.DataFrame(feature_records)

    logger.debug(
        "Feature vector prepared for batch prediction",
        extra={
            "event": "feature_vector",
            "endpoint": "/predict/batch",
            "batch_size": batch_size,
            "features": feature_records,
        },
    )

    try:
        predictions = predictor.predict_batch(X)

    except Exception as exc:
        logger.error(
            "Prediction failed",
            extra={
                "event": "prediction_failed",
                "model_version": settings.model_version,
                "endpoint": "/predict/batch",
                "method": request.method,
                "batch_size": batch_size,
                "status_code": 500,
                "error_type": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    logger.info(
        "Prediction served successfully",
        extra={
            "event": "prediction_successful",
            "model_version": settings.model_version,
            "endpoint": "/predict/batch",
            "method": request.method,
            "batch_size": batch_size,
            "status_code": 200,
            "latency_ms": prediction_latency_ms.get(),
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
    )
