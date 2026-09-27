import json
import logging
import sys
from datetime import UTC, datetime

from prodml.logging.context import correlation_id


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    _OPTIONAL_FIELDS = (
        "event",
        "model_version",
        "endpoint",
        "method",
        "duration_ms",
        "latency_ms",
        "batch_size",
        "error_type",
        "status_code",
        "reason",
        "function",
        "features",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record into a JSON object."""

        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(),
        }

        for field in self._OPTIONAL_FIELDS:
            value = getattr(record, field, None)

            if value is not None:
                log_entry[field] = value

        return json.dumps(log_entry)


def configure_logging() -> None:
    """Configure application-wide structured JSON logging."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
