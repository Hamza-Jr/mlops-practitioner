import pandas as pd
from fastapi import APIRouter, Depends

from prodml.api.dependencies import get_predictor
from prodml.api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.models.predictor import WildfirePredictor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    predictor: WildfirePredictor = Depends(get_predictor),  # noqa: B008
) -> PredictionResponse:
    """Return a prediction for a single wildfire observation."""

    X = pd.DataFrame(
        [
            {
                "ndvi": request.ndvi,
                "lst": request.lst,
                "burned_area": request.burned_area,
            }
        ]
    )

    result = predictor.predict_one(X)

    return PredictionResponse(
        class_name=result["class"],
        probability=float(result["probability"]),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    predictor: WildfirePredictor = Depends(get_predictor),  # noqa: B008
) -> BatchPredictionResponse:
    """Return predictions for a batch of wildfire observations."""

    X = pd.DataFrame(
        [
            {
                "ndvi": item.ndvi,
                "lst": item.lst,
                "burned_area": item.burned_area,
            }
            for item in request.instances
        ]
    )

    predictions = predictor.predict_batch(X)

    return BatchPredictionResponse(
        predictions=predictions,
    )
