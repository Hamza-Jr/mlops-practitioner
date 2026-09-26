from fastapi import APIRouter, Depends

from prodml.api.dependencies import get_predictor
from prodml.api.schemas.health import HealthResponse
from prodml.models.predictor import WildfirePredictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    predictor: WildfirePredictor = Depends(get_predictor),  # noqa: B008
) -> HealthResponse:
    """Return the health status of the prediction service."""

    return HealthResponse(status="healthy")
