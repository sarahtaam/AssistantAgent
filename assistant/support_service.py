"""
assistant/support_service.py — Proactive support escalation.

UX principle: the agent detects a problem, then PROPOSES options via chat.
It never creates a ticket without the client's consent.

Escalation rules (faithful to the original):
  - System-triggered events (e.g. a frontend reporting a failed face
    verification) -> always escalate immediately.
  - Urgent complaint types (payment failed, biometric verification
    failed) -> escalate on first mention, since these block the client
    from completing an action right now.
  - Lower-urgency complaints (general/technical) -> wait for a second
    mention before proposing a ticket, to avoid over-triggering on a
    single offhand remark.

Note: the original also routes tickets to a specific boutique/appointment
subsystem. That's dropped here (out of scope for this portfolio repo —
see README) in favor of a simple ticket log the admin can review.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Tuple

from sqlalchemy import text

from config import SUPPORT_PHONE
from db import get_db_connection

logger = logging.getLogger(__name__)


class ComplaintType(str, Enum):
    FACE_VERIFY_FAILED = "FACE_VERIFY_FAILED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    CONTRACT_ISSUE = "CONTRACT_ISSUE"
    TECHNICAL_ISSUE = "TECHNICAL_ISSUE"
    GENERAL_COMPLAINT = "GENERAL_COMPLAINT"


_LABELS = {
    ComplaintType.FACE_VERIFY_FAILED: "an identity verification issue",
    ComplaintType.PAYMENT_FAILED: "a payment issue",
    ComplaintType.CONTRACT_ISSUE: "a contract/activation issue",
    ComplaintType.TECHNICAL_ISSUE: "a technical issue",
    ComplaintType.GENERAL_COMPLAINT: "an issue",
}

# Types that block the client from completing an action right now ->
# escalate on first mention rather than waiting for repetition.
_IMMEDIATE_TYPES = frozenset([
    ComplaintType.FACE_VERIFY_FAILED,
    ComplaintType.PAYMENT_FAILED,
    ComplaintType.CONTRACT_ISSUE,
])

_FACE_KW = frozenset(["face id", "vérification", "identité", "biométrie", "caméra", "selfie", "verification failed"])
_PAYMENT_KW = frozenset(["paiement échoué", "carte refusée", "payment failed", "transaction échouée", "impossible de payer"])
_CONTRACT_KW = frozenset(["contrat non activé", "esim ne s'active", "activation échouée", "profil esim"])
_TECH_KW = frozenset(["bug", "erreur technique", "ne fonctionne pas", "ne marche pas", "plantage", "bloqué"])


def detect_complaint_type(message: str) -> Optional[ComplaintType]:
    lower = message.lower()
    if any(kw in lower for kw in _FACE_KW):
        return ComplaintType.FACE_VERIFY_FAILED
    if any(kw in lower for kw in _PAYMENT_KW):
        return ComplaintType.PAYMENT_FAILED
    if any(kw in lower for kw in _CONTRACT_KW):
        return ComplaintType.CONTRACT_ISSUE
    if any(kw in lower for kw in _TECH_KW):
        return ComplaintType.TECHNICAL_ISSUE
    if any(w in lower for w in ("réclamation", "plainte", "complaint", "problème")):
        return ComplaintType.GENERAL_COMPLAINT
    return None


def should_escalate(
    message: str, session_complaint_count: int = 0, triggered_by_system: bool = False,
) -> Tuple[bool, Optional[ComplaintType]]:
    ctype = detect_complaint_type(message)

    if triggered_by_system:
        return True, ctype or ComplaintType.TECHNICAL_ISSUE

    if ctype is None:
        return False, None

    if ctype in _IMMEDIATE_TYPES:
        return True, ctype

    return session_complaint_count >= 2, ctype  # softer types: wait for repetition


def build_first_mention_response(prenom: str, ctype: ComplaintType) -> dict:
    label = _LABELS.get(ctype, "an issue")
    return {
        "response": (
            f"I'm sorry to hear you're running into {label}, {prenom}. "
            "Could you tell me a bit more about what's happening? "
            "If it keeps happening I'll open a support ticket for you right away."
        ),
        "options": [],
    }


def build_proposal_response(prenom: str, complaint_type: ComplaintType, session_id: str = "") -> dict:
    ctx = _LABELS.get(complaint_type, "an issue")
    return {
        "response": (
            f"I'm sorry, {prenom} — it looks like you're dealing with {ctx}. "
            f"I can open a support ticket right now so our team follows up, "
            f"or you can reach us directly at {SUPPORT_PHONE}. How would you like to proceed?"
        ),
        "options": [
            {"id": "create_ticket", "label": "\U0001F3AB Open a support ticket",
             "type": "quick_reply", "message": "Yes, please open a support ticket"},
            {"id": "call_support", "label": f"\U0001F4DE Call {SUPPORT_PHONE}",
             "type": "phone", "url": f"tel:{SUPPORT_PHONE}"},
        ],
    }


def auto_create__ticket(client: dict, complaint_type: ComplaintType) -> dict:
    conn = get_db_connection()
    try:
        ticket_id = conn.execute(
            text("""INSERT INTO tickets (client_id, complaint_type, status)
                    VALUES (:cid, :ctype, 'OPEN') RETURNING id"""),
            {"cid": client["id"], "ctype": complaint_type.value},
        ).scalar_one()

        conn.execute(text("""
            INSERT INTO admin_notifications (type, client_id, reference_id, message, created_at, read)
            VALUES ('COMPLAINT', :cid, :ref, :msg, CURRENT_TIMESTAMP, 0)
        """), {
            "cid": client["id"], "ref": ticket_id,
            "msg": f"{complaint_type.value} | {client.get('prenom')} {client.get('nom')} | Ticket #{ticket_id}",
        })
        conn.commit()

        logger.info("Ticket #%s created for client %d (%s)", ticket_id, client["id"], complaint_type.value)
        return {
            "success": True,
            "ticket_id": ticket_id,
            "response": f"Done — I've opened ticket #{ticket_id} for you. Our support team will follow up shortly.",
        }
    except Exception as exc:
        conn.rollback()
        logger.error("auto_create__ticket failed: %s", exc)
        return {"success": False, "response": f"I couldn't open a ticket automatically. Please contact {SUPPORT_PHONE}."}
    finally:
        conn.close()
