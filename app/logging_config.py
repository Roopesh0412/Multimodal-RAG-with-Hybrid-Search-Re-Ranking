"""
Structured enterprise logging configuration.
Provides JSON formatting for production telemetry and readable console logs.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_payload.update(record.extra)
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_payload)


def configure_logging(log_level: str = "INFO", json_logs: bool = False) -> logging.Logger:
    """Configures the root and application loggers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
    root_logger.addHandler(handler)

    # Silence chatty third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger("rag_enterprise")
