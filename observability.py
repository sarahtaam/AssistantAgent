"""
observability.py — Logging, request tracing and error tracking.

- Logs: one JSON object per line in production (LOG_FORMAT=json), so a log
  aggregator can index fields; readable text locally. Every line carries the
  request id of the request that produced it.
- Request ids: taken from an incoming X-Request-ID header when it looks sane
  (so a load balancer's id flows through), otherwise generated; echoed back
  in the response header for support tickets and bug reports.
- Access log + metrics: one line and one Prometheus sample per request,
  labelled by route template (/scoring/{client_id}), never the raw path.
- Errors: reported to Sentry when SENTRY_DSN is set — without request
  bodies, local variables or user details, which would carry personal data.
"""
import contextvars
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone

from starlette.types import ASGIApp, Message, Receive, Scope, Send

import config
import metrics

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_access_logger = logging.getLogger("access")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    _EXTRA_FIELDS = ("method", "route", "status", "duration_ms")

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for field in self._EXTRA_FIELDS:
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    if config.LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(config.LOG_LEVEL)
    # The access middleware below replaces uvicorn's access log.
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True


def setup_error_tracking() -> None:
    if not config.SENTRY_DSN:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        environment=config.APP_ENV,
        traces_sample_rate=config.SENTRY_TRACES_SAMPLE_RATE,
        send_default_pii=False,
        max_request_body_size="never",     # chat messages are personal data
        include_local_variables=False,     # so are the variables holding them
    )
    logging.getLogger(__name__).info("Error tracking enabled")


class RequestContextMiddleware:
    """Pure ASGI middleware (no BaseHTTPMiddleware: it would buffer
    streaming responses and break contextvars in sync endpoints)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or {}).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration = time.perf_counter() - start
            route = scope.get("route")
            route_label = getattr(route, "path", "unmatched")
            method = scope["method"]
            metrics.HTTP_REQUESTS.labels(method=method, route=route_label, status=str(status)).inc()
            metrics.HTTP_LATENCY.labels(method=method, route=route_label).observe(duration)
            _access_logger.info(
                "%s %s %d %.0fms", method, route_label, status, duration * 1000,
                extra={"method": method, "route": route_label, "status": status,
                       "duration_ms": round(duration * 1000, 1)},
            )
            request_id_var.reset(token)
