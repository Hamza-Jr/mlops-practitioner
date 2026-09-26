import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from prodml.logging.context import correlation_id

logger = logging.getLogger("prodml.api")


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to each HTTP request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        token = correlation_id.set(request_id)

        start_time = time.perf_counter()
        status_code = 500

        logger.info(
            "http_request_started",
            extra={
                "endpoint": request.url.path,
                "method": request.method,
            },
        )

        try:
            response = await call_next(request)
            status_code = response.status_code

            response.headers["X-Request-ID"] = request_id

            return response

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            extra = {
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            }

            if status_code >= 500:
                logger.error("http_request_completed", extra=extra)
            elif status_code >= 400:
                logger.warning("http_request_completed", extra=extra)
            else:
                logger.info("http_request_completed", extra=extra)

            correlation_id.reset(token)
