import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prodml.api.middleware.correlation import CorrelationMiddleware
from prodml.api.routes.health import router as health_router
from prodml.api.routes.metadata import router as metadata_router
from prodml.api.routes.predict import router as predict_router
from prodml.logging.logging_config import configure_logging
from prodml.models.onnx_model import ONNXWildfireModel
from prodml.models.predictor import WildfirePredictor
from prodml.utils.config import settings
from prodml.utils.hashing import calculate_sha256

configure_logging()

logger = logging.getLogger("prodml.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model once during application startup."""

    logger.info(
        "application_started",
        extra={
            "model_version": settings.model_version,
        },
    )

    model = ONNXWildfireModel(settings.onnx_model_path)
    app.state.predictor = WildfirePredictor(model)
    app.state.artifact_hash = calculate_sha256(settings.onnx_model_path)

    logger.info(
        "model_loaded",
        extra={
            "model_version": settings.model_version,
        },
    )

    yield

    logger.info(
        "application_shutdown",
        extra={
            "model_version": settings.model_version,
        },
    )

    app.state.predictor = None
    app.state.artifact_hash = None


app = FastAPI(
    title="Wildfire Classification API",
    description="API for wildfire classification using the production model.",
    version=settings.model_version,
    lifespan=lifespan,
)

app.add_middleware(CorrelationMiddleware)

app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(predict_router)
