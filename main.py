"""
main.py — Assistant Agent API (billing & collections assistant).

This is the trimmed public version: the conversational agent, scoring,
and payment negotiation endpoints. The original system also has an
admin back-office (dashboards, appointment scheduling, ticket triage)
that's intentionally left out here — it's standard CRUD, not part of
the AI differentiator this project demonstrates.

Endpoints                                        Auth
──────────────────────────────────────────────────────────────────────
GET  /health                                → —        liveness check
GET  /ready                                 → —        DB + Redis reachable
GET  /client/assistant/auto-session/{id}    → client   session + personalized welcome
POST /client/assistant/chat                 → client*  message -> agent response
POST /client/assistant/option-click         → client*  click on an interactive option
POST /client/assistant/confirm              → client   confirm a payment plan
POST /client/assistant/support-ticket       → client   open a support ticket
GET  /scoring/{client_id}                   → staff    risk score + explanation
GET  /ml/score-combine/{client_id}          → staff    hybrid rules+ML score

client = bearer token for that client only (see auth.py)
*      = also rate-limited per client, since it can trigger a paid LLM call

Errors: handlers log the full traceback server-side and return a generic
500 body. Internal exception text (SQL, file paths) is deliberately never
echoed back to the caller.
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

import config
from assistant.agent.orchestrator import get_agent
from assistant.memory.short_term import SessionOwnershipError
from assistant.support_service import ComplaintType
from auth import Principal, ensure_same_client, require_client, require_staff
from db import get_db_connection
from rate_limit import limit_llm_calls
from recouvrement.predict import score_combine
from recouvrement.scoring import calculer_score_client
import redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_INTERNAL_ERROR = "Internal error."


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.validate_settings()
    yield


app = FastAPI(
    title="Assistant Agent",
    description="AI customer assistant for telecom billing and collections: risk scoring, payment-plan negotiation, and support escalation.",
    version="1.1",
    lifespan=lifespan,
)

# Bearer tokens travel in the Authorization header, not cookies, so
# credentials mode is off. Only the listed front-end origins may call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(SessionOwnershipError)
async def _session_ownership_handler(_request: Request, _exc: SessionOwnershipError):
    return JSONResponse(status_code=403, content={"detail": "This session belongs to another client."})


# `client_id` in bodies is optional and only cross-checked against the token
# (kept so existing front-ends keep working); the token decides who you are.

class ChatRequest(BaseModel):
    session_id: str = Field(max_length=100)
    client_id: Optional[int] = None
    message: str = Field(min_length=1, max_length=config.MAX_MESSAGE_LENGTH)


class ConfirmPlanRequest(BaseModel):
    session_id: str = Field(max_length=100)
    client_id: Optional[int] = None
    option_id: int


class OptionClickRequest(BaseModel):
    session_id: str = Field(max_length=100)
    client_id: Optional[int] = None
    option_id: str = Field(max_length=50)


class SupportTicketRequest(BaseModel):
    session_id: str = Field(max_length=100)
    client_id: Optional[int] = None
    complaint_type: ComplaintType


def _internal_error(name: str) -> HTTPException:
    logger.exception("%s failed", name)
    return HTTPException(status_code=500, detail=_INTERNAL_ERROR)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    checks = {"database": True}
    try:
        conn = get_db_connection()
        try:
            conn.execute(text("SELECT 1"))
        finally:
            conn.close()
    except Exception:
        logger.exception("Readiness: database unreachable")
        checks["database"] = False

    redis_ok = redis_client.ping()
    if redis_ok is not None:
        checks["redis"] = redis_ok

    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "unavailable", **checks})


@app.get("/client/assistant/auto-session/{client_id}")
def auto_session(client_id: int, principal: Principal = Depends(require_client)):
    ensure_same_client(principal, client_id)
    try:
        return get_agent().auto_session(client_id)
    except Exception:
        raise _internal_error("auto_session")


@app.post("/client/assistant/chat")
def chat(req: ChatRequest, principal: Principal = Depends(limit_llm_calls)):
    client_id = ensure_same_client(principal, req.client_id)
    try:
        return get_agent().process_message(req.session_id, client_id, req.message)
    except SessionOwnershipError:
        raise
    except Exception:
        raise _internal_error("chat")


@app.post("/client/assistant/option-click")
def option_click(req: OptionClickRequest, principal: Principal = Depends(limit_llm_calls)):
    client_id = ensure_same_client(principal, req.client_id)
    try:
        return get_agent().handle_option_click(req.session_id, client_id, req.option_id)
    except SessionOwnershipError:
        raise
    except Exception:
        raise _internal_error("option_click")


@app.post("/client/assistant/confirm")
def confirm(req: ConfirmPlanRequest, principal: Principal = Depends(require_client)):
    client_id = ensure_same_client(principal, req.client_id)
    try:
        return get_agent().confirm_plan(req.session_id, client_id, req.option_id)
    except SessionOwnershipError:
        raise
    except Exception:
        raise _internal_error("confirm")


@app.post("/client/assistant/support-ticket")
def support_ticket(req: SupportTicketRequest, principal: Principal = Depends(require_client)):
    client_id = ensure_same_client(principal, req.client_id)
    try:
        return get_agent().create_support_ticket(req.session_id, client_id, req.complaint_type)
    except SessionOwnershipError:
        raise
    except Exception:
        raise _internal_error("support_ticket")


@app.get("/scoring/{client_id}")
def scoring(client_id: int, _staff: Principal = Depends(require_staff)):
    try:
        return calculer_score_client(client_id)
    except Exception:
        raise _internal_error("scoring")


@app.get("/ml/score-combine/{client_id}")
def ml_score_combine(client_id: int, _staff: Principal = Depends(require_staff)):
    try:
        rules_score = calculer_score_client(client_id)["score"]
        return score_combine(client_id, rules_score)
    except Exception:
        raise _internal_error("ml_score_combine")
