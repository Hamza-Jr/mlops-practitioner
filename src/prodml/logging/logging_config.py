import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format application log records as single-line JSON objects."""

    _OPTIONAL_FIELDS = (
        "request_id",
        "model_version",
        "endpoint",
        "method",
        "duration_ms",
        "batch_size",
        "error_type",
        "status_code",
        "reason",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record into a JSON object."""

        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
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
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
