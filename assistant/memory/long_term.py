"""
assistant/memory/long_term.py — Persistent memory (database-backed).
Stores confirmed payment plans, interaction summaries, and admin
notifications. SQLite in this demo; swap DATABASE_URL for Postgres
in production without touching this module — the queries here only use
SQL both dialects accept (CURRENT_TIMESTAMP, INSERT ... RETURNING).
"""
import contextvars
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Set

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from config import COMPANY_NAME, CURRENCY_SYMBOL, SUPPORT_PHONE
from db import get_db_connection
from privacy import mask_contact

logger = logging.getLogger(__name__)

# Confirmation emails are sent off the request path: a slow or unreachable
# SMTP server must not hold up the client's "plan confirmed" response.
# Pending sends still finish on a graceful shutdown (executor threads are
# joined at interpreter exit).
_mail_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mailer")
_pending_emails: Set[Future] = set()


class PlanAlreadyConfirmedError(Exception):
    """The client already has a CONFIRMED plan (unique index violation)."""


def drain_email_queue(timeout: float = 10.0) -> None:
    """Waits for queued confirmation emails (used by tests)."""
    wait(list(_pending_emails), timeout=timeout)


# ── Payment plans ────────────────────────────────────────────────────

def get_active_plan(client_id: int) -> Optional[Dict]:
    """The client's CONFIRMED plan, if any."""
    conn = get_db_connection()
    try:
        row = conn.execute(text("""
            SELECT id, session_id, plan_data, summary, status, created_at
            FROM payment_plans WHERE client_id = :cid AND status = 'CONFIRMED'
            ORDER BY created_at DESC LIMIT 1
        """), {"cid": client_id}).fetchone()
        return dict(row._mapping) if row else None
    finally:
        conn.close()


def save_payment_plan(client_id: int, session_id: str, plan_data: dict, summary: str) -> Optional[int]:
    """Saves a CONFIRMED plan and queues the confirmation email. Returns the
    plan id, or None on a database error. Raises PlanAlreadyConfirmedError
    if the client already has a confirmed plan."""
    conn = get_db_connection()
    try:
        row = conn.execute(text("""
            INSERT INTO payment_plans (client_id, session_id, plan_data, summary, status, created_at)
            VALUES (:cid, :sid, :data, :summary, 'CONFIRMED', CURRENT_TIMESTAMP)
            RETURNING id
        """), {
            "cid": client_id,
            "sid": session_id,
            "data": json.dumps(plan_data, ensure_ascii=False),
            "summary": summary,
        }).fetchone()
        conn.commit()
        plan_id = row[0]
        logger.info("Payment plan #%d created for client %d", plan_id, client_id)
    except IntegrityError:
        conn.rollback()
        raise PlanAlreadyConfirmedError(client_id)
    except Exception:
        conn.rollback()
        logger.exception("save_payment_plan failed for client %d", client_id)
        return None
    finally:
        conn.close()

    # copy_context: the mailer's log lines keep this request's request id.
    future = _mail_executor.submit(contextvars.copy_context().run, _envoyer_email_plan, client_id, plan_data, summary)
    _pending_emails.add(future)
    future.add_done_callback(_pending_emails.discard)
    return plan_id


def _envoyer_email_plan(client_id: int, plan_data: dict, summary: str) -> None:
    """Sends a payment-plan confirmation email (simulated by default)."""
    from recouvrement.actions import envoyer_email_simulation
    conn = get_db_connection()
    try:
        row = conn.execute(
            text("SELECT email_or_phone, prenom, nom FROM users WHERE id = :cid"), {"cid": client_id}
        ).fetchone()
        if not row:
            return

        dest, prenom, nom = row[0], row[1] or "", row[2] or ""
        versements = plan_data.get("versements", [])
        details = "\n".join(
            f"  - {v.get('label', '')}: {CURRENCY_SYMBOL}{v.get('montant', 0):.2f} - {v.get('date', '')}"
            for v in versements
        )

        corps = f"""Hello {prenom} {nom},

Your payment plan has been confirmed.

Summary:
{summary}

Installment details:
{details}

Please respect the payment dates to avoid suspension of your line.

Cordially,
{COMPANY_NAME}"""

        envoyer_email_simulation(dest, f"Payment plan confirmed - {COMPANY_NAME}", corps)
        # The mailer logs the actual outcome (really sent vs. simulated),
        # so don't claim delivery here.
        logger.info("Plan confirmation email handed to mailer -> %s", mask_contact(dest))
    except Exception:
        logger.exception("Plan email error for client %d", client_id)
    finally:
        conn.close()


def get_client_plans(client_id: int) -> List[Dict]:
    conn = get_db_connection()
    try:
        rows = conn.execute(text("""
            SELECT id, session_id, plan_data, summary, status, created_at
            FROM payment_plans WHERE client_id = :cid ORDER BY created_at DESC
        """), {"cid": client_id}).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        conn.close()


def update_plan_status(plan_id: int, status: str) -> bool:
    conn = get_db_connection()
    try:
        conn.execute(text("""
            UPDATE payment_plans SET status = :status, updated_at = CURRENT_TIMESTAMP WHERE id = :pid
        """), {"status": status, "pid": plan_id})
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        logger.error("update_plan_status error: %s", exc)
        return False
    finally:
        conn.close()


# ── Interaction history ─────────────────────────────────────────────

def save_interaction_summary(client_id: int, session_id: str, summary: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute(text("""
            INSERT INTO client_interactions (client_id, session_id, summary, created_at)
            VALUES (:cid, :sid, :summary, CURRENT_TIMESTAMP)
        """), {"cid": client_id, "sid": session_id, "summary": summary})
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("save_interaction_summary error: %s", exc)
    finally:
        conn.close()


def get_client_long_term_history(client_id: int, limit: int = 5) -> str:
    """Returns a readable text block summarizing the client's recent
    interactions - injected into the system prompt for long-term memory."""
    conn = get_db_connection()
    try:
        rows = conn.execute(text("""
            SELECT summary, created_at FROM client_interactions
            WHERE client_id = :cid ORDER BY created_at DESC LIMIT :lim
        """), {"cid": client_id, "lim": limit}).fetchall()

        if not rows:
            return "No previous interaction history."

        return "\n".join(f"[{row.created_at}] {row.summary}" for row in rows)
    finally:
        conn.close()


# ── Admin notifications ─────────────────────────────────────────────

def notify_admin(client_id: int, reference_id: Optional[int], message: str, notif_type: str = "PAYMENT_PLAN") -> None:
    conn = get_db_connection()
    try:
        conn.execute(text("""
            INSERT INTO admin_notifications (type, client_id, reference_id, message, created_at, read)
            VALUES (:type, :cid, :ref, :msg, CURRENT_TIMESTAMP, 0)
        """), {"type": notif_type, "cid": client_id, "ref": reference_id, "msg": message})
        conn.commit()
        logger.info("Admin notification created: type=%s, client=%d", notif_type, client_id)
    except Exception as exc:
        conn.rollback()
        logger.error("notify_admin error: %s", exc)
    finally:
        conn.close()


def get_admin_notifications(unread_only: bool = False) -> List[Dict]:
    conn = get_db_connection()
    try:
        where = "WHERE n.read = 0" if unread_only else ""
        rows = conn.execute(text(f"""
            SELECT n.id, n.type, n.client_id, u.nom, u.prenom,
                   n.reference_id, n.message, n.created_at, n.read
            FROM admin_notifications n
            LEFT JOIN users u ON n.client_id = u.id
            {where}
            ORDER BY n.created_at DESC LIMIT 100
        """)).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        conn.close()


def mark_notification_read(notif_id: int) -> None:
    conn = get_db_connection()
    try:
        conn.execute(text("UPDATE admin_notifications SET read = 1 WHERE id = :nid"), {"nid": notif_id})
        conn.commit()
    finally:
        conn.close()
