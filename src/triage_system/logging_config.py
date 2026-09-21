"""Privacy-safe request logging for the HTTP service."""

import logging
import time

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("triage_system.http")


async def log_request(request: Request, call_next) -> Response:
    """Log request metadata only; never read or emit request bodies."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "request failed method=%s path=%s duration_ms=%.1f",
            request.method, request.url.path, elapsed_ms,
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request completed method=%s path=%s status=%s duration_ms=%.1f",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response
