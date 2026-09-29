"""
tests/test_session_store.py — Both session backends must behave the same.
The Redis backend runs against fakeredis, so no server is needed.
"""
import time

import fakeredis
import pytest

from assistant.memory.short_term import MAX_TURNS, InMemorySessionStore, RedisSessionStore


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return InMemorySessionStore()
    return RedisSessionStore(fakeredis.FakeRedis(decode_responses=True))


class TestSessionStore:

    def test_session_records_its_owner(self, store):
        sid = store.create_session(7)
        assert store.owner_of(sid) == 7
        assert store.owner_of("s_unknown") is None

    def test_session_ids_are_random_not_derived_from_client_id(self, store):
        sid = store.create_session(7)
        assert not sid.startswith("s_7_")
        assert len(sid.removeprefix("s_")) == 32  # full uuid4, 122 random bits

    def test_messages_round_trip_and_are_trimmed(self, store):
        sid = store.create_session(7)
        for i in range(MAX_TURNS + 5):
            store.add_message(sid, "user", f"m{i}")
        history = store.get_llm_history(sid)
        assert len(history) == MAX_TURNS
        assert history[-1] == {"role": "user", "content": f"m{MAX_TURNS + 4}"}

    def test_plans_context_and_complaints(self, store):
        sid = store.create_session(7)
        plans = [{"id": 1, "label": "2-month plan", "total": 99.5}]
        store.set_proposed_plans(sid, plans)
        store.set_context(sid, "pending_plan", plans[0])
        store.set_confirmed_plan(sid, plans[0])
        assert store.get_proposed_plans(sid) == plans
        assert store.get_context(sid, "pending_plan") == plans[0]
        assert store.get_confirmed_plan(sid) == plans[0]
        assert store.increment_complaint_count(sid) == 1
        assert store.increment_complaint_count(sid) == 2
        store.reset_complaint_count(sid)
        assert store.get_complaint_count(sid) == 0

    def test_mutating_an_unknown_session_is_a_no_op(self, store):
        store.add_message("s_unknown", "user", "hi")
        store.set_proposed_plans("s_unknown", [{"id": 1}])
        assert store.increment_complaint_count("s_unknown") == 0
        assert store.get_session("s_unknown") is None

    def test_find_active_session_is_per_client(self, store):
        sid_a = store.create_session(1)
        store.create_session(2)
        assert store.find_active_session(1)["session_id"] == sid_a
        assert store.find_active_session(3) is None

    def test_returned_session_is_a_copy(self, store):
        sid = store.create_session(7)
        store.get_session(sid)["messages"].append({"role": "user", "content": "sneaky"})
        assert store.get_llm_history(sid) == []

    def test_close_session(self, store):
        sid = store.create_session(7)
        store.close_session(sid)
        assert store.get_session(sid) is None


class TestExpiry:

    def test_memory_sessions_expire_and_are_purged(self):
        store = InMemorySessionStore()
        store.ttl_seconds = 0.05
        sid = store.create_session(7)
        time.sleep(0.1)
        assert store.get_session(sid) is None
        assert store.find_active_session(7) is None

        store.create_session(8)
        time.sleep(0.1)
        assert store.purge_expired() == 1
        assert store._sessions == {}

    def test_memory_store_purges_on_create(self, monkeypatch):
        store = InMemorySessionStore()
        store.ttl_seconds = 0.01
        for client_id in range(50):
            store.create_session(client_id)
        time.sleep(0.05)
        monkeypatch.setattr("assistant.memory.short_term._PURGE_INTERVAL_SECONDS", 0)
        store.create_session(99)
        assert len(store._sessions) == 1

    def test_redis_sessions_carry_a_ttl(self):
        fake = fakeredis.FakeRedis(decode_responses=True)
        store = RedisSessionStore(fake)
        sid = store.create_session(7)
        ttl = fake.ttl(f"novatel:session:{sid}")
        assert 0 < ttl <= store.ttl_seconds

    def test_redis_sessions_are_shared_between_store_instances(self):
        fake = fakeredis.FakeRedis(decode_responses=True)
        worker_1, worker_2 = RedisSessionStore(fake), RedisSessionStore(fake)
        sid = worker_1.create_session(7)
        worker_1.add_message(sid, "user", "hello")
        assert worker_2.get_llm_history(sid) == [{"role": "user", "content": "hello"}]
        assert worker_2.owner_of(sid) == 7
