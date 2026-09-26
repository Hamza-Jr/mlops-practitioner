import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
            "event": "application_started",
            "model_version": settings.model_version,
        },
    )

    try:
        model = ONNXWildfireModel(settings.onnx_model_path)
        app.state.predictor = WildfirePredictor(model)
        app.state.artifact_hash = calculate_sha256(settings.onnx_model_path)

    except Exception as exc:
        logger.error(
            "model_load_failed",
            extra={
                "event": "model_load_failed",
                "model_version": settings.model_version,
                "error_type": type(exc).__name__,
            },
        )
        raise

    logger.info(
        "model_loaded",
        extra={
            "event": "model_loaded",
            "model_version": settings.model_version,
        },
    )

    yield

    logger.info(
        "application_shutdown",
        extra={
            "event": "application_shutdown",
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Log rejected requests and return a safe validation response."""

    logger.error(
        "Request validation rejected",
        extra={
            "event": "validation_rejected",
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": 422,
            "error_type": "RequestValidationError",
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed.",
        },
    )


app.add_middleware(CorrelationMiddleware)

app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(predict_router)
