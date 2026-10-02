"""
assistant/memory/short_term.py — Session memory (short-term).
Sliding window: MAX_TURNS messages. Sessions expire after SESSION_TTL_SECONDS
of inactivity.

Two interchangeable backends behind one interface:
  - RedisSessionStore (REDIS_URL set): shared by every worker and replica,
    survives app restarts, expiry handled by Redis TTLs.
  - InMemorySessionStore (default): zero infra for local development; a
    single process only. Expired sessions are purged as new ones are created.

Every session records the client it was created for. Callers must check
ownership with `owner_of()` before acting on a session id supplied by a
request — see AssistantAgent._require_owned_session().
"""
import copy
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from config import SESSION_TTL_SECONDS
from redis_client import get_redis

logger = logging.getLogger(__name__)

MAX_TURNS = 12
_PURGE_INTERVAL_SECONDS = 60
_REDIS_PREFIX = "novatel:session:"
_REDIS_CLIENT_INDEX = "novatel:client-session:"


class SessionOwnershipError(Exception):
    """A request used a session id that belongs to a different client."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Shared behaviour. Backends implement _load, _save_new, _mutate,
    _delete and find_active_session."""

    ttl_seconds = SESSION_TTL_SECONDS

    # ── Backend primitives ────────────────────────────────────────────
    def _load(self, session_id: str) -> Optional[dict]:
        raise NotImplementedError

    def _save_new(self, session: dict) -> None:
        raise NotImplementedError

    def _mutate(self, session_id: str, fn: Callable[[dict], Any]) -> Any:
        """Applies fn to the stored session atomically; returns fn's result,
        or None if the session doesn't exist."""
        raise NotImplementedError

    def _delete(self, session_id: str) -> None:
        raise NotImplementedError

    def find_active_session(self, client_id: int) -> Optional[dict]:
        """Returns the most recently active session for a client, or None."""
        raise NotImplementedError

    # ── Session lifecycle ─────────────────────────────────────────────
    def create_session(self, client_id: int) -> str:
        sid = f"s_{uuid.uuid4().hex}"
        now = _now_iso()
        self._save_new({
            "session_id": sid,
            "client_id": client_id,
            "created_at": now,
            "last_active": now,
            "messages": [],
            "proposed_plans": [],
            "confirmed_plan": None,
            "context": {},
            "complaint_count": 0,
        })
        logger.info("Session created: %s (client %d)", sid, client_id)
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        """Returns the session and marks it active (slides its expiry)."""
        def touch(session: dict) -> dict:
            session["last_active"] = _now_iso()
            return copy.deepcopy(session)
        return self._mutate(session_id, touch)

    def owner_of(self, session_id: str) -> Optional[int]:
        session = self._load(session_id)
        return session["client_id"] if session else None

    def close_session(self, session_id: str) -> None:
        self._delete(session_id)
        logger.info("Session closed: %s", session_id)

    # ── Messages ──────────────────────────────────────────────────────
    def add_message(self, session_id: str, role: str, content: str) -> None:
        def append(session: dict) -> bool:
            session["messages"].append({"role": role, "content": content, "ts": _now_iso()})
            session["messages"] = session["messages"][-MAX_TURNS:]
            session["last_active"] = _now_iso()
            return True
        if self._mutate(session_id, append) is None:
            logger.warning("add_message - unknown session: %s", session_id)

    def get_llm_history(self, session_id: str) -> List[Dict]:
        """Returns history in the format expected by the Groq/OpenAI API."""
        session = self._load(session_id)
        if not session:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in session["messages"]]

    # ── Payment plans ─────────────────────────────────────────────────
    def set_proposed_plans(self, session_id: str, plans: List[dict]) -> None:
        self._mutate(session_id, lambda s: s.__setitem__("proposed_plans", plans))

    def get_proposed_plans(self, session_id: str) -> List[dict]:
        return (self._load(session_id) or {}).get("proposed_plans", [])

    def set_confirmed_plan(self, session_id: str, plan: dict) -> None:
        self._mutate(session_id, lambda s: s.__setitem__("confirmed_plan", plan))

    def get_confirmed_plan(self, session_id: str) -> Optional[dict]:
        return (self._load(session_id) or {}).get("confirmed_plan")

    def claim_confirmation(self, session_id: str, plan: dict) -> bool:
        """Atomically marks `plan` as this session's confirmed plan. Returns
        False if the session already has one (or doesn't exist), so a double
        click or a retried request can't confirm twice."""
        def claim(session: dict) -> bool:
            if session.get("confirmed_plan"):
                return False
            session["confirmed_plan"] = plan
            return True
        return bool(self._mutate(session_id, claim))

    def release_confirmation(self, session_id: str) -> None:
        """Undoes claim_confirmation() when saving the plan failed."""
        self.set_confirmed_plan(session_id, None)

    # ── Free-form context ─────────────────────────────────────────────
    def set_context(self, session_id: str, key: str, value: Any) -> None:
        self._mutate(session_id, lambda s: s["context"].__setitem__(key, value))

    def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        return (self._load(session_id) or {}).get("context", {}).get(key, default)

    # ── Complaint counter (drives support escalation thresholds) ──────
    def increment_complaint_count(self, session_id: str) -> int:
        def incr(session: dict) -> int:
            session["complaint_count"] += 1
            return session["complaint_count"]
        return self._mutate(session_id, incr) or 0

    def get_complaint_count(self, session_id: str) -> int:
        return (self._load(session_id) or {}).get("complaint_count", 0)

    def reset_complaint_count(self, session_id: str) -> None:
        self._mutate(session_id, lambda s: s.__setitem__("complaint_count", 0))


class InMemorySessionStore(SessionStore):
    """Single-process store. Thread-safe via RLock."""

    def __init__(self) -> None:
        self._sessions: Dict[str, dict] = {}
        self._expires: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._last_purge = time.monotonic()

    def _alive(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._expires[session_id] <= time.monotonic():
            self._sessions.pop(session_id, None)
            self._expires.pop(session_id, None)
            return None
        return session

    def _load(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._alive(session_id)
            return copy.deepcopy(session) if session else None

    def _save_new(self, session: dict) -> None:
        with self._lock:
            self._maybe_purge()
            self._sessions[session["session_id"]] = session
            self._expires[session["session_id"]] = time.monotonic() + self.ttl_seconds

    def _mutate(self, session_id: str, fn: Callable[[dict], Any]) -> Any:
        with self._lock:
            session = self._alive(session_id)
            if session is None:
                return None
            result = fn(session)
            self._expires[session_id] = time.monotonic() + self.ttl_seconds
            return result

    def _delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._expires.pop(session_id, None)

    def find_active_session(self, client_id: int) -> Optional[dict]:
        with self._lock:
            candidates = [
                s for sid, s in list(self._sessions.items())
                if s["client_id"] == client_id and self._alive(sid)
            ]
            if not candidates:
                return None
            return copy.deepcopy(max(candidates, key=lambda s: s["last_active"]))

    def purge_expired(self) -> int:
        """Removes inactive sessions."""
        now = time.monotonic()
        with self._lock:
            expired = [sid for sid, exp in self._expires.items() if exp <= now]
            for sid in expired:
                self._sessions.pop(sid, None)
                self._expires.pop(sid, None)
            self._last_purge = now
        if expired:
            logger.info("Expired sessions purged: %d", len(expired))
        return len(expired)

    def _maybe_purge(self) -> None:
        if time.monotonic() - self._last_purge >= _PURGE_INTERVAL_SECONDS:
            self.purge_expired()


class RedisSessionStore(SessionStore):
    """Sessions as JSON strings with a sliding TTL. Updates use optimistic
    locking (WATCH/MULTI) so concurrent workers never lose a write."""

    _MAX_RETRIES = 10

    def __init__(self, client) -> None:
        self._redis = client

    @staticmethod
    def _key(session_id: str) -> str:
        return _REDIS_PREFIX + session_id

    def _load(self, session_id: str) -> Optional[dict]:
        raw = self._redis.get(self._key(session_id))
        return json.loads(raw) if raw else None

    def _save_new(self, session: dict) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._key(session["session_id"]), json.dumps(session), ex=self.ttl_seconds)
        pipe.set(_REDIS_CLIENT_INDEX + str(session["client_id"]), session["session_id"], ex=self.ttl_seconds)
        pipe.execute()

    def _mutate(self, session_id: str, fn: Callable[[dict], Any]) -> Any:
        from redis.exceptions import WatchError

        key = self._key(session_id)
        for _ in range(self._MAX_RETRIES):
            with self._redis.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if not raw:
                        return None
                    session = json.loads(raw)
                    result = fn(session)
                    pipe.multi()
                    pipe.set(key, json.dumps(session), ex=self.ttl_seconds)
                    pipe.set(_REDIS_CLIENT_INDEX + str(session["client_id"]), session_id, ex=self.ttl_seconds)
                    pipe.execute()
                    return result
                except WatchError:
                    continue
        raise RuntimeError(f"Session {session_id} update kept conflicting; giving up.")

    def _delete(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def find_active_session(self, client_id: int) -> Optional[dict]:
        sid = self._redis.get(_REDIS_CLIENT_INDEX + str(client_id))
        if not sid:
            return None
        session = self._load(sid)
        if not session or session["client_id"] != client_id:
            return None
        return session


def create_session_store() -> SessionStore:
    client = get_redis()
    if client is not None:
        logger.info("Session store: Redis")
        return RedisSessionStore(client)
    logger.info("Session store: in-memory (single process only — set REDIS_URL for production)")
    return InMemorySessionStore()


session_memory: SessionStore = create_session_store()
