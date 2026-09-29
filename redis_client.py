"""
redis_client.py — Shared Redis connection (sessions + rate limiting).

Returns None when REDIS_URL is unset, so callers fall back to in-process
storage for local development.
"""
from typing import Optional

from config import REDIS_URL

_client = None


def get_redis():
    global _client
    if not REDIS_URL:
        return None
    if _client is None:
        import redis

        _client = redis.Redis.from_url(
            REDIS_URL, decode_responses=True, socket_timeout=5, socket_connect_timeout=5,
        )
    return _client


def ping() -> Optional[bool]:
    """True/False when Redis is configured, None when it isn't."""
    client = get_redis()
    if client is None:
        return None
    try:
        return bool(client.ping())
    except Exception:
        return False
