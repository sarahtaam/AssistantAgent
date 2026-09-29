"""
db.py — Database connection layer.

Original version connected to a production Postgres instance with a
company-specific schema. This version defaults to SQLite so the project runs
with `pip install -r requirements.txt && python seed.py && uvicorn main:app`
and nothing else. Set DATABASE_URL to postgresql+psycopg://... for production.

Queries in this codebase stick to SQL that SQLite and Postgres both accept.
Date arithmetic (days overdue, "last N months") is done in Python with the
helpers below instead of dialect-specific functions like julianday().
"""
import os
from datetime import date, datetime
from typing import Optional, Union

from sqlalchemy import create_engine

from config import DATABASE_URL

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _ensure_sqlite_dir(url: str) -> None:
    """Create the SQLite file's parent directory if it's missing.

    Git doesn't track empty directories, so `data/` doesn't exist in a fresh
    clone — and SQLite won't create it, it just fails to open the database.
    Doing this here covers every entry point (seed.py, the API, training).
    """
    if not url.startswith("sqlite"):
        return
    path = url.split("///", 1)[-1]
    if path in ("", ":memory:") or path.startswith(":memory:"):
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


_ensure_sqlite_dir(DATABASE_URL)

_is_sqlite = DATABASE_URL.startswith("sqlite")
_engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # Drop connections the server closed (DB restart, idle timeout) instead
    # of failing the next request with them.
    pool_pre_ping=not _is_sqlite,
)


def get_db_connection():
    """Returns a SQLAlchemy connection. Caller is responsible for closing it."""
    return _engine.connect()


def get_engine():
    return _engine


def migrate() -> None:
    """Applies every pending Alembic migration (`alembic upgrade head`)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(os.path.join(_PROJECT_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(_PROJECT_ROOT, "migrations"))
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")


# ── Portable date helpers ─────────────────────────────────────────────────

def to_date(value: Union[date, datetime, str, None]) -> Optional[date]:
    """Normalizes a DATE/TIMESTAMP column value to a date.

    Postgres returns date/datetime objects; SQLite returns ISO strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def to_datetime(value: Union[date, datetime, str, None]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def days_between(start: Union[date, datetime, str], end: Union[date, datetime, str]) -> int:
    """Whole days from start to end (negative if end is earlier)."""
    return (to_date(end) - to_date(start)).days


def months_ago(months: int, today: Optional[date] = None) -> date:
    """Same day-of-month `months` calendar months back, clamped to month end
    (e.g. 31 March - 1 month -> 28/29 February)."""
    today = today or date.today()
    month_index = today.year * 12 + (today.month - 1) - months
    year, month = divmod(month_index, 12)
    month += 1
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last_day = (next_month - date(year, month, 1)).days
    return date(year, month, min(today.day, last_day))
