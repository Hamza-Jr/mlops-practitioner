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

router = APIRouter()

logger = logging.getLogger("prodml.api")


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    payload: PredictionRequest,
    predictor: WildfirePredictor = Depends(get_predictor),  # noqa: B008
) -> PredictionResponse:
    """Return a prediction for a single wildfire observation."""

    request_id = request.state.request_id

    logger.info(
        "prediction_requested",
        extra={
            "request_id": request_id,
            "model_version": settings.model_version,
            "endpoint": "/predict",
            "method": request.method,
        },
    )

    X = pd.DataFrame(
        [
            {
                "ndvi": payload.ndvi,
                "lst": payload.lst,
                "burned_area": payload.burned_area,
            }
        ]
    )

    try:
        result = predictor.predict_one(X)

    except Exception as exc:
        logger.error(
            "prediction_failed",
            extra={
                "request_id": request_id,
                "model_version": settings.model_version,
                "endpoint": "/predict",
                "error_type": type(exc).__name__,
                "status_code": 500,
                "method": request.method,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    logger.info(
        "prediction_successful",
        extra={
            "request_id": request_id,
            "model_version": settings.model_version,
            "endpoint": "/predict",
            "method": request.method,
            "status_code": 200,
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

    request_id = request.state.request_id

    logger.info(
        "prediction_requested",
        extra={
            "request_id": request_id,
            "model_version": settings.model_version,
            "endpoint": "/predict/batch",
            "method": request.method,
            "batch_size": len(payload.instances),
        },
    )

    X = pd.DataFrame(
        [
            {
                "ndvi": item.ndvi,
                "lst": item.lst,
                "burned_area": item.burned_area,
            }
            for item in payload.instances
        ]
    )

    try:
        predictions = predictor.predict_batch(X)

    except Exception as exc:
        logger.error(
            "prediction_failed",
            extra={
                "request_id": request_id,
                "model_version": settings.model_version,
                "endpoint": "/predict/batch",
                "method": request.method,
                "batch_size": len(payload.instances),
                "error_type": type(exc).__name__,
                "status_code": 500,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    logger.info(
        "prediction_successful",
        extra={
            "request_id": request_id,
            "model_version": settings.model_version,
            "endpoint": "/predict/batch",
            "method": request.method,
            "batch_size": len(payload.instances),
            "status_code": 200,
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
    )
