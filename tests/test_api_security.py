"""
tests/test_api_security.py — Authentication, authorization, session
ownership, input validation, rate limiting and CORS, through the real API.
"""
import time

import jwt
import pytest

import config
import rate_limit
from rate_limit import Limit, RateLimiter

CLIENT_A = 1
CLIENT_B = 2


def _open_session(client, bearer, client_id):
    resp = client.get(f"/client/assistant/auto-session/{client_id}", headers=bearer(client_id))
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


class TestAuthentication:

    @pytest.mark.parametrize("method,path,body", [
        ("get", f"/client/assistant/auto-session/{CLIENT_A}", None),
        ("post", "/client/assistant/chat", {"session_id": "x", "message": "hi"}),
        ("post", "/client/assistant/option-click", {"session_id": "x", "option_id": "1"}),
        ("post", "/client/assistant/confirm", {"session_id": "x", "option_id": 1}),
        ("post", "/client/assistant/support-ticket", {"session_id": "x", "complaint_type": "PAYMENT_FAILED"}),
        ("get", f"/scoring/{CLIENT_A}", None),
        ("get", f"/ml/score-combine/{CLIENT_A}", None),
    ])
    def test_every_data_endpoint_requires_a_token(self, client, method, path, body):
        resp = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"

    def test_health_is_public(self, client):
        assert client.get("/health").status_code == 200

    def test_ready_reports_database(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["database"] is True

    def test_token_signed_with_another_key_is_rejected(self, client):
        forged = jwt.encode({"sub": str(CLIENT_A), "exp": int(time.time()) + 60}, "not-the-real-secret" * 3, "HS256")
        resp = client.get(f"/client/assistant/auto-session/{CLIENT_A}", headers={"Authorization": f"Bearer {forged}"})
        assert resp.status_code == 401

    def test_expired_token_is_rejected(self, client, bearer):
        resp = client.get(f"/client/assistant/auto-session/{CLIENT_A}", headers=bearer(CLIENT_A, ttl_seconds=-120))
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Token expired."

    def test_token_without_expiry_is_rejected(self, client):
        token = jwt.encode({"sub": str(CLIENT_A)}, config.JWT_SECRET, "HS256")
        resp = client.get(f"/client/assistant/auto-session/{CLIENT_A}", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_unsigned_alg_none_token_is_rejected(self, client):
        token = jwt.encode({"sub": str(CLIENT_A), "exp": int(time.time()) + 60}, None, algorithm="none")
        resp = client.get(f"/client/assistant/auto-session/{CLIENT_A}", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401


class TestClientIsolation:

    def test_client_can_open_own_session(self, client, bearer):
        resp = client.get(f"/client/assistant/auto-session/{CLIENT_A}", headers=bearer(CLIENT_A))
        assert resp.status_code == 200
        assert resp.json()["session_id"].startswith("s_")

    def test_client_cannot_open_another_clients_session(self, client, bearer):
        resp = client.get(f"/client/assistant/auto-session/{CLIENT_B}", headers=bearer(CLIENT_A))
        assert resp.status_code == 403

    def test_body_client_id_must_match_token(self, client, bearer):
        resp = client.post("/client/assistant/chat", headers=bearer(CLIENT_A),
                           json={"session_id": "x", "client_id": CLIENT_B, "message": "hello"})
        assert resp.status_code == 403

    def test_client_cannot_read_risk_scores(self, client, bearer):
        assert client.get(f"/scoring/{CLIENT_A}", headers=bearer(CLIENT_A)).status_code == 403
        assert client.get(f"/ml/score-combine/{CLIENT_A}", headers=bearer(CLIENT_A)).status_code == 403

    def test_staff_can_read_any_risk_score(self, client, bearer):
        resp = client.get(f"/scoring/{CLIENT_B}", headers=bearer("ops-user", "staff"))
        assert resp.status_code == 200
        assert resp.json()["client_id"] == CLIENT_B

    def test_staff_cannot_chat_as_a_client(self, client, bearer):
        resp = client.post("/client/assistant/chat", headers=bearer("ops-user", "staff"),
                           json={"session_id": "x", "message": "hello"})
        assert resp.status_code == 403


class TestSessionOwnership:

    @pytest.mark.parametrize("path,body", [
        ("/client/assistant/chat", {"message": "je veux payer"}),
        ("/client/assistant/option-click", {"option_id": "1"}),
        ("/client/assistant/confirm", {"option_id": 1}),
        ("/client/assistant/support-ticket", {"complaint_type": "PAYMENT_FAILED"}),
    ])
    def test_cannot_use_another_clients_session(self, client, bearer, path, body):
        victim_session = _open_session(client, bearer, CLIENT_A)
        resp = client.post(path, headers=bearer(CLIENT_B), json={"session_id": victim_session, **body})
        assert resp.status_code == 403

    def test_owner_can_chat_and_confirm_in_own_session(self, client, bearer):
        session_id = _open_session(client, bearer, CLIENT_A)
        chat = client.post("/client/assistant/chat", headers=bearer(CLIENT_A),
                           json={"session_id": session_id, "message": "je veux un plan de paiement"})
        assert chat.status_code == 200, chat.text
        body = chat.json()
        assert body["session_id"] == session_id
        assert body["options"], "client 1 has unpaid invoices, so plans should be offered"

        confirm = client.post("/client/assistant/confirm", headers=bearer(CLIENT_A),
                              json={"session_id": session_id, "option_id": body["options"][0]["id"]})
        assert confirm.status_code == 200
        assert confirm.json()["success"] is True
        assert isinstance(confirm.json()["plan_id"], int)

    def test_unknown_session_is_replaced_with_a_fresh_one(self, client, bearer):
        resp = client.post("/client/assistant/chat", headers=bearer(CLIENT_A),
                           json={"session_id": "s_does_not_exist", "message": "bonjour"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] != "s_does_not_exist"


class TestInputValidation:

    def test_unknown_complaint_type_is_a_422_not_a_500(self, client, bearer):
        session_id = _open_session(client, bearer, CLIENT_A)
        resp = client.post("/client/assistant/support-ticket", headers=bearer(CLIENT_A),
                           json={"session_id": session_id, "complaint_type": "NOT_A_TYPE"})
        assert resp.status_code == 422

    def test_valid_complaint_type_opens_a_ticket(self, client, bearer):
        session_id = _open_session(client, bearer, CLIENT_A)
        resp = client.post("/client/assistant/support-ticket", headers=bearer(CLIENT_A),
                           json={"session_id": session_id, "complaint_type": "PAYMENT_FAILED"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert isinstance(resp.json()["ticket_id"], int)

    def test_oversized_message_is_rejected(self, client, bearer):
        resp = client.post("/client/assistant/chat", headers=bearer(CLIENT_A),
                           json={"session_id": "x", "message": "a" * (config.MAX_MESSAGE_LENGTH + 1)})
        assert resp.status_code == 422


class TestRateLimiting:

    def test_chat_is_limited_per_client(self, client, bearer, monkeypatch):
        monkeypatch.setattr(rate_limit, "llm_limiter", RateLimiter([Limit(2, 60)]))
        body = {"session_id": "x", "message": "bonjour"}

        for _ in range(2):
            assert client.post("/client/assistant/chat", headers=bearer(CLIENT_A), json=body).status_code == 200
        blocked = client.post("/client/assistant/chat", headers=bearer(CLIENT_A), json=body)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1

        # Another client has its own budget.
        assert client.post("/client/assistant/chat", headers=bearer(CLIENT_B), json=body).status_code == 200

    def test_option_click_shares_the_llm_budget(self, client, bearer, monkeypatch):
        monkeypatch.setattr(rate_limit, "llm_limiter", RateLimiter([Limit(1, 60)]))
        client.post("/client/assistant/chat", headers=bearer(CLIENT_A), json={"session_id": "x", "message": "bonjour"})
        resp = client.post("/client/assistant/option-click", headers=bearer(CLIENT_A),
                           json={"session_id": "x", "option_id": "1"})
        assert resp.status_code == 429


class TestCors:

    def test_allowed_origin_gets_cors_headers(self, client):
        resp = client.options("/client/assistant/chat", headers={
            "Origin": "https://app.novatel.example", "Access-Control-Request-Method": "POST",
        })
        assert resp.headers.get("access-control-allow-origin") == "https://app.novatel.example"
        assert "access-control-allow-credentials" not in resp.headers

    def test_unknown_origin_gets_no_cors_headers(self, client):
        resp = client.options("/client/assistant/chat", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
        })
        assert "access-control-allow-origin" not in resp.headers
