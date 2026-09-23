from contextlib import asynccontextmanager

from fastapi import FastAPI

from prodml.api.routes.health import router as health_router
from prodml.api.routes.metadata import router as metadata_router
from prodml.models.onnx_model import ONNXWildfireModel
from prodml.models.predictor import WildfirePredictor
from prodml.utils.config import settings
from prodml.utils.hashing import calculate_sha256


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model once during application startup."""

    model = ONNXWildfireModel(settings.onnx_model_path)
    app.state.predictor = WildfirePredictor(model)
    app.state.artifact_hash = calculate_sha256(settings.onnx_model_path)

    yield

    app.state.predictor = None
    app.state.artifact_hash = None


app = FastAPI(
    title="Wildfire Classification API",
    description="API for wildfire classification using the production model.",
    version=settings.model_version,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(metadata_router)
