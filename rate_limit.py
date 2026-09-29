"""
rate_limit.py — Per-client fixed-window rate limiting.

Guards the endpoints that can trigger a paid LLM call. Counters live in
Redis when REDIS_URL is set (shared by every worker and replica), otherwise
in process memory for local development.

Keyed by the authenticated caller, not by IP: behind a load balancer or a
mobile carrier NAT, many legitimate clients share one IP.
"""
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, status

from auth import Principal, require_client
from config import LLM_RATE_LIMITS
from redis_client import get_redis

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


@dataclass(frozen=True)
class Limit:
    count: int
    period_seconds: int


def parse_limits(spec: str) -> List[Limit]:
    """'20/minute;300/day' -> [Limit(20, 60), Limit(300, 86400)]"""
    limits = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        count, _, unit = part.partition("/")
        unit = unit.strip().lower().rstrip("s")
        if unit not in _PERIODS or not count.strip().isdigit():
            raise ValueError(f"Invalid rate limit {part!r}; expected e.g. '20/minute'.")
        limits.append(Limit(int(count), _PERIODS[unit]))
    return limits


class RateLimiter:

    def __init__(self, limits: List[Limit], redis_client=None) -> None:
        self.limits = limits
        self._redis = redis_client
        self._counts: Dict[str, Tuple[int, float]] = {}
        self._lock = threading.Lock()

    def hit(self, bucket: str, key: str) -> Optional[int]:
        """Counts one request. Returns None if allowed, else the number of
        seconds until the caller may retry."""
        now = time.time()
        retry_after = None
        for limit in self.limits:
            window = int(now // limit.period_seconds)
            counter_key = f"novatel:rl:{bucket}:{key}:{limit.period_seconds}:{window}"
            count = self._incr(counter_key, limit.period_seconds)
            if count > limit.count:
                wait = math.ceil((window + 1) * limit.period_seconds - now)
                retry_after = max(retry_after or 0, wait, 1)
        return retry_after

    def _incr(self, key: str, ttl: int) -> int:
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl + 1)
            count, _ = pipe.execute()
            return int(count)

        now = time.time()
        with self._lock:
            if len(self._counts) > 10_000:
                self._counts = {k: v for k, v in self._counts.items() if v[1] > now}
            count, expires = self._counts.get(key, (0, now + ttl))
            count += 1
            self._counts[key] = (count, expires)
            return count

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


llm_limiter = RateLimiter(parse_limits(LLM_RATE_LIMITS), get_redis())


def limit_llm_calls(principal: Principal = Depends(require_client)) -> Principal:
    """FastAPI dependency: authenticates a client and applies the LLM limits."""
    retry_after = llm_limiter.hit("llm", principal.subject)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(retry_after)},
        )
    return principal
