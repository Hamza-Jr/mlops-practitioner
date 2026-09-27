from fastapi import APIRouter, Request

from prodml.api.schemas.metadata import MetadataResponse
from prodml.utils.config import settings

router = APIRouter()


@router.get("/metadata", response_model=MetadataResponse)
def get_metadata(request: Request) -> MetadataResponse:
    """Return metadata for the production model."""

    return MetadataResponse(
        model_version=settings.model_version,
        training_date=settings.training_date,
        feature_names=list(settings.feature_names),
        framework=settings.framework,
        artifact_hash=request.app.state.artifact_hash,
    )
