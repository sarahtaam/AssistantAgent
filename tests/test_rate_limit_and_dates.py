"""
tests/test_rate_limit_and_dates.py — Rate limiter backends and the portable
date helpers that replaced SQLite-only SQL functions.
"""
from datetime import date, datetime

import fakeredis
import pytest

from db import days_between, months_ago, to_date
from rate_limit import Limit, RateLimiter, parse_limits


class TestParseLimits:

    def test_parses_multiple_limits(self):
        assert parse_limits("20/minute; 300/day") == [Limit(20, 60), Limit(300, 86400)]

    def test_accepts_plural_units(self):
        assert parse_limits("5/seconds") == [Limit(5, 1)]

    @pytest.mark.parametrize("spec", ["20", "x/minute", "20/fortnight"])
    def test_rejects_garbage(self, spec):
        with pytest.raises(ValueError):
            parse_limits(spec)


@pytest.mark.parametrize("backend", ["memory", "redis"])
class TestRateLimiter:

    def _limiter(self, backend, limits):
        redis = fakeredis.FakeRedis(decode_responses=True) if backend == "redis" else None
        return RateLimiter(limits, redis)

    def test_allows_up_to_the_limit_then_blocks(self, backend):
        limiter = self._limiter(backend, [Limit(3, 60)])
        assert [limiter.hit("llm", "1") for _ in range(3)] == [None, None, None]
        retry_after = limiter.hit("llm", "1")
        assert retry_after is not None and 1 <= retry_after <= 60

    def test_keys_are_independent(self, backend):
        limiter = self._limiter(backend, [Limit(1, 60)])
        assert limiter.hit("llm", "1") is None
        assert limiter.hit("llm", "2") is None
        assert limiter.hit("llm", "1") is not None

    def test_strictest_limit_wins(self, backend):
        limiter = self._limiter(backend, [Limit(100, 60), Limit(2, 86400)])
        limiter.hit("llm", "1")
        limiter.hit("llm", "1")
        assert limiter.hit("llm", "1") > 60  # blocked by the daily limit


class TestDateHelpers:

    @pytest.mark.parametrize("value", [
        date(2026, 3, 5), datetime(2026, 3, 5, 14, 30), "2026-03-05", "2026-03-05 14:30:00",
    ])
    def test_to_date_accepts_sqlite_and_postgres_values(self, value):
        assert to_date(value) == date(2026, 3, 5)

    def test_days_between(self):
        assert days_between("2026-03-01", date(2026, 3, 11)) == 10
        assert days_between(date(2026, 3, 11), "2026-03-01") == -10

    @pytest.mark.parametrize("today,months,expected", [
        (date(2026, 9, 29), 6, date(2026, 3, 29)),
        (date(2026, 3, 31), 1, date(2026, 2, 28)),
        (date(2028, 3, 31), 1, date(2028, 2, 29)),
        (date(2026, 1, 15), 1, date(2025, 12, 15)),
        (date(2026, 12, 31), 12, date(2025, 12, 31)),
    ])
    def test_months_ago(self, today, months, expected):
        assert months_ago(months, today) == expected
