"""
config.py — Single source of truth for company branding, business constants,
and deployment settings.

In the original codebase these values (company name, support line, currency,
booking URL) were hardcoded inside prompt templates and the RAG fallback
knowledge base. Centralizing them here means the entire agent can be
re-skinned for a different company/domain by editing this one file.
"""
import os
from typing import List

from dotenv import load_dotenv

# Load .env before anything reads os.environ. Real environment variables
# (e.g. set by docker-compose or CI) take precedence over the file.
load_dotenv()


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# ── Environment ───────────────────────────────────────────────────────────
# "production" turns on the fail-fast checks in validate_settings().
APP_ENV = os.getenv("APP_ENV", "development").lower()
IS_PRODUCTION = APP_ENV == "production"

# ── Branding ──────────────────────────────────────────────────────────────
COMPANY_NAME = os.getenv("COMPANY_NAME", "NovaTel")
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "1234")
BOOKING_URL = os.getenv("BOOKING_URL", "https://demo.novatel.example/book-appointment")
CURRENCY = os.getenv("CURRENCY", "USD")
CURRENCY_SYMBOL = os.getenv("CURRENCY_SYMBOL", "$")

# ── Database ──────────────────────────────────────────────────────────────
# SQLite by default so the project runs with zero external setup. Point
# DATABASE_URL at Postgres (postgresql+psycopg://...) for production.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/novatel_agent.db")

# ── Sessions & rate limiting ──────────────────────────────────────────────
# Unset -> in-process memory (single worker, lost on restart; fine for dev).
# Set -> sessions and rate-limit counters are shared by every worker/replica.
REDIS_URL = os.getenv("REDIS_URL", "")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 60 * 60)))

# Limits on the endpoints that can trigger a paid LLM call, per client.
# Format: "<count>/<second|minute|hour|day>", several separated by ";".
LLM_RATE_LIMITS = os.getenv("LLM_RATE_LIMITS", "20/minute;300/day")

# ── Authentication ────────────────────────────────────────────────────────
# The API does not issue credentials itself: it verifies JWTs issued by the
# operator's identity provider. Use JWT_SECRET for HS256, or JWT_PUBLIC_KEY
# (PEM) for RS256/ES256.
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", "").replace("\\n", "\n")
JWT_ALGORITHMS = _csv(os.getenv("JWT_ALGORITHMS", "HS256"))
JWT_ISSUER = os.getenv("JWT_ISSUER", "")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "")
MIN_JWT_SECRET_LENGTH = 32

# ── CORS ──────────────────────────────────────────────────────────────────
# Comma-separated front-end origins. Empty -> no cross-origin browser access.
CORS_ALLOWED_ORIGINS = _csv(os.getenv("CORS_ALLOWED_ORIGINS", ""))

# ── LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
# Upper bounds on how long one LLM call may hold a request: per HTTP attempt,
# and in total across retries. Past the total, the agent answers with its
# deterministic fallback instead of keeping the client waiting.
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "10"))
LLM_TOTAL_TIMEOUT_SECONDS = float(os.getenv("LLM_TOTAL_TIMEOUT_SECONDS", "15"))

# ── Observability ─────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# "json" (one JSON object per line, for log aggregators) or "text".
LOG_FORMAT = os.getenv("LOG_FORMAT", "json" if IS_PRODUCTION else "text").lower()
# /metrics is served only when this is set, and only with this bearer token.
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")
# Optional error tracking (Sentry or a compatible service).
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0"))

# ── Risk-tiering thresholds (business rules, not proprietary — a common
#    industry pattern for graduated collections strategies) ────────────────
SCORE_TIERS = {
    # (min, max): (max_installments, interval_days, advance_pct, label, emoji)
    # Boundaries and labels match recouvrement/scoring.py's classification exactly.
    (85, 100): (4, 30, 0.00, "EXCELLENT", "\U0001F7E2"),
    (65, 84): (3, 30, 0.00, "FIABLE", "\U0001F535"),
    (45, 64): (2, 30, 0.20, "MOYEN", "\U0001F7E1"),
    (25, 44): (2, 7, 0.30, "RISQUE", "\U0001F7E0"),
    (0, 24): (1, 2, 1.00, "CRITIQUE", "\U0001F534"),
}

MAX_REALISTIC_AMOUNT = 100_000.0  # guardrail: block hallucinated amounts above this


def validate_settings() -> None:
    """Refuses to start with a configuration that would be unsafe to serve.

    Called once at app startup. Auth is checked in every environment (without
    a key no request can be authenticated); the rest only in production.
    """
    problems: List[str] = []

    if not JWT_SECRET and not JWT_PUBLIC_KEY:
        problems.append("JWT_SECRET or JWT_PUBLIC_KEY must be set.")
    uses_hmac = any(alg.startswith("HS") for alg in JWT_ALGORITHMS)
    if uses_hmac and not JWT_SECRET:
        problems.append("JWT_ALGORITHMS includes an HS* algorithm but JWT_SECRET is empty.")

    if IS_PRODUCTION:
        if uses_hmac and JWT_SECRET and len(JWT_SECRET) < MIN_JWT_SECRET_LENGTH:
            problems.append(f"JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters.")
        if not REDIS_URL:
            problems.append("REDIS_URL must be set (sessions and rate limits must be shared across workers).")
        if "*" in CORS_ALLOWED_ORIGINS:
            problems.append("CORS_ALLOWED_ORIGINS must list explicit origins, not '*'.")
        if DATABASE_URL.startswith("sqlite"):
            problems.append("DATABASE_URL points at SQLite; use Postgres in production.")
        if METRICS_TOKEN and len(METRICS_TOKEN) < 24:
            problems.append("METRICS_TOKEN must be at least 24 characters.")

    if problems:
        raise RuntimeError("Invalid configuration:\n  - " + "\n  - ".join(problems))
