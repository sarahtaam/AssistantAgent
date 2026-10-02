"""
metrics.py — Prometheus metrics, served at /metrics (see main.py).

With several uvicorn workers, set PROMETHEUS_MULTIPROC_DIR to an empty,
writable directory (the Docker image does) so every worker's numbers are
aggregated; without it each worker would report only its own.

Label values are always from a small fixed set (route templates, outcome
names) — never ids or user input, which would blow up cardinality and
could leak personal data.
"""
import os

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled.", ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency.", ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40),
)
LLM_REQUESTS = Counter(
    "llm_requests_total", "Calls to the LLM provider.", ["outcome"],  # ok | error | not_configured
)
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds", "LLM call latency, retries included.",
    buckets=(0.5, 1, 2, 4, 8, 15, 25),
)
DEGRADED_REPLIES = Counter(
    "degraded_replies_total", "Chat replies served with deterministic wording because the LLM was unavailable.",
)
GUARDRAIL_BLOCKS = Counter(
    "guardrail_blocks_total", "Messages or responses blocked by a guardrail.", ["guard", "reason"],
)
RATE_LIMITED = Counter(
    "rate_limited_requests_total", "Requests rejected by the per-client rate limiter.",
)
PLANS_CONFIRMED = Counter(
    "payment_plans_confirmed_total", "Payment plans confirmed by clients.",
)


def render() -> tuple[bytes, str]:
    """Current metrics in the Prometheus text format, all workers included."""
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
