import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("prodml.api")


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and log HTTP request completion."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        request.state.request_id = request_id

        start_time = time.perf_counter()
        response = None
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code

            return response

        except Exception as exc:
            logger.error(
                "http_request_failed",
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "status_code": status_code,
                    "error_type": type(exc).__name__,
                },
            )

            raise

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            request.state.duration_ms = duration_ms

            if response is not None:
                response.headers["X-Request-ID"] = request_id

            extra = {
                "request_id": request_id,
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            }

            if status_code >= 500:
                logger.error(
                    "http_request_completed",
                    extra=extra,
                )
            elif status_code >= 400:
                logger.warning(
                    "http_request_completed",
                    extra=extra,
                )
            else:
                logger.info(
                    "http_request_completed",
                    extra=extra,
                )
