"""
Shared test setup.

Settings are read from the environment at import time, so they're set here,
before any project module is imported. DATABASE_URL / REDIS_URL from the
environment win (CI runs the same suite against Postgres + Redis); otherwise
tests get a throwaway SQLite file and in-memory sessions.
"""
import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="novatel-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_dir}/test.db")
os.environ["APP_ENV"] = "test"
os.environ["JWT_SECRET"] = "test-secret-that-is-long-enough-for-hs256-0123456789"
os.environ["JWT_ALGORITHMS"] = "HS256"
os.environ["JWT_ISSUER"] = ""
os.environ["JWT_AUDIENCE"] = ""
os.environ["CORS_ALLOWED_ORIGINS"] = "https://app.novatel.example"
os.environ["LLM_RATE_LIMITS"] = "1000/minute"
# Never let a developer's .env key make tests call the real LLM.
os.environ["GROQ_API_KEY"] = ""

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def seeded_db():
    import seed

    seed.seed()


@pytest.fixture
def client(seeded_db):
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from rate_limit import llm_limiter

    llm_limiter.reset()
    yield
    llm_limiter.reset()


@pytest.fixture
def bearer():
    """bearer(3) -> Authorization header for client 3; bearer("ops", "staff") for staff."""
    from auth import issue_token

    def _headers(subject, role="client", ttl_seconds=3600) -> dict:
        return {"Authorization": f"Bearer {issue_token(str(subject), role, ttl_seconds)}"}

    return _headers
