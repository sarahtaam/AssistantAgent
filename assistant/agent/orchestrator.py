"""
assistant/agent/orchestrator.py — Core agent loop.

This is the piece that ties everything together: client context assembly,
RAG retrieval (with graceful fallback), intent classification, guardrails,
and payment negotiation. See the README for the full architecture diagram.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Optional, Tuple

from sqlalchemy import Date, bindparam, text

import metrics
from config import COMPANY_NAME, CURRENCY_SYMBOL
from db import days_between, get_db_connection, months_ago
from recouvrement.scoring import calculer_score_client

from assistant.rag.retriever import get_retriever
from assistant.llm.groq_client import get_groq_client
from assistant.llm.prompt_templates import (
    PromptTemplates, WELCOME_TEMPLATE, offline_reply, SESSION_EXPIRED_MSG,
)
from assistant.llm.output_guard import sanitize_user_input, validate_and_sanitize
from assistant.memory.short_term import SessionOwnershipError, session_memory
from assistant.memory.long_term import (
    PlanAlreadyConfirmedError, get_active_plan, save_payment_plan, get_client_long_term_history,
    save_interaction_summary, notify_admin, get_client_plans,
)
from assistant.agent.negotiator import (
    get_negotiator, INTENT_OFF_TOPIC, INTENT_GREETING, INTENT_SUPPORT_REQUEST,
)
from assistant.support_service import (
    build_first_mention_response,
    build_proposal_response,
    ComplaintType,
    auto_create__ticket,
    should_escalate,
)

logger = logging.getLogger(__name__)


# ── Data access ─────────────────────────────────────────────────────────

def _fetch_client_info(client_id: int) -> Optional[dict]:
    conn = get_db_connection()
    try:
        row = conn.execute(
            text("SELECT id, nom, prenom, email_or_phone, telephone FROM users WHERE id = :cid"),
            {"cid": client_id},
        ).fetchone()
        return dict(row._mapping) if row else None
    finally:
        conn.close()


def _fetch_unpaid_invoices(client_id: int) -> List[dict]:
    conn = get_db_connection()
    try:
        esim = conn.execute(text("""
            SELECT f.id, f.montant, f.date_echeance, 'ESIM' AS type_ligne
            FROM factures f JOIN contrats c ON f.contrat_id = c.id
            WHERE c.user_id = :cid AND f.statut IN ('EN_ATTENTE', 'EN_RETARD')
            ORDER BY f.date_echeance ASC
        """), {"cid": client_id}).fetchall()
        sim = conn.execute(text("""
            SELECT f.id, f.montant, f.date_echeance, 'SIM' AS type_ligne
            FROM factures f JOIN contrats_sim c ON f.contrat_sim_id = c.id
            WHERE c.user_id = :cid AND f.statut IN ('EN_ATTENTE', 'EN_RETARD')
            ORDER BY f.date_echeance ASC
        """), {"cid": client_id}).fetchall()
        today = date.today()
        invoices = []
        for r in list(esim) + list(sim):
            inv = dict(r._mapping)
            inv["jours_retard"] = max(days_between(inv["date_echeance"], today), 0)
            invoices.append(inv)
        return invoices
    finally:
        conn.close()


def _fetch_payment_history(client_id: int, months: int = 6) -> List[dict]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            text("""
                SELECT f.id, f.montant, f.date_echeance, f.date_paiement
                FROM factures f JOIN contrats c ON f.contrat_id = c.id
                WHERE c.user_id = :cid AND f.statut = 'PAYEE'
                  AND f.date_paiement >= :since
                ORDER BY f.date_paiement DESC LIMIT 10
            """).bindparams(bindparam("since", type_=Date)),
            {"cid": client_id, "since": months_ago(months)},
        ).fetchall()
        history = []
        for r in rows:
            p = dict(r._mapping)
            late_days = days_between(p["date_echeance"], p["date_paiement"])
            p["statut_paiement"] = "A_TEMPS" if late_days <= 0 else "EN_RETARD"
            p["jours_retard"] = max(late_days, 0)
            history.append(p)
        return history
    except Exception:
        logger.exception("_fetch_payment_history failed")
        return []
    finally:
        conn.close()


def _fetch_contract_status(client_id: int) -> dict:
    conn = get_db_connection()
    try:
        esim = conn.execute(text("""
            SELECT c.id, c.statut, c.numero_contrat, 'ESIM' AS type_ligne,
                   pe.statut AS statut_profil
            FROM contrats c
            LEFT JOIN profils_esim pe ON pe.contrat_id = c.id
            WHERE c.user_id = :cid AND c.statut = 'ACTIVE'
        """), {"cid": client_id}).fetchall()

        sim = conn.execute(text("""
            SELECT c.id, c.statut, c.numero_contrat, 'SIM' AS type_ligne,
                   s.statut AS statut_profil
            FROM contrats_sim c
            LEFT JOIN souscriptions_sim s ON s.contrat_sim_id = c.id
            WHERE c.user_id = :cid AND c.statut = 'ACTIVE'
        """), {"cid": client_id}).fetchall()

        all_c = [dict(r._mapping) for r in list(esim) + list(sim)]

        STATUTS_ACTIFS = {"ACTIF", "ACTIVE"}
        STATUTS_SUSPENDUS = {"SUSPENDU", "SUSPENDED"}
        STATUTS_DESACTIVES = {"DESACTIVE", "DESACTIVEE", "DESACTIVÉ"}
        STATUTS_ATTENTE = {"EN_ATTENTE", "EN_ATTENTE_ACTIVATION", "EN_ATTENTE_PAIEMENT", "EN_ATTENTE_VALIDATION"}

        def _count(type_ligne, statuts):
            return len([c for c in all_c if c["type_ligne"] == type_ligne and c.get("statut_profil") in statuts])

        tous_suspendus = _count("ESIM", STATUTS_SUSPENDUS) + _count("SIM", STATUTS_SUSPENDUS)

        return {
            "total": len(all_c),
            "esim_actives": _count("ESIM", STATUTS_ACTIFS),
            "esim_suspendus": _count("ESIM", STATUTS_SUSPENDUS),
            "esim_desactives": _count("ESIM", STATUTS_DESACTIVES),
            "esim_attente": _count("ESIM", STATUTS_ATTENTE),
            "sim_actives": _count("SIM", STATUTS_ACTIFS),
            "sim_suspendus": _count("SIM", STATUTS_SUSPENDUS),
            "sim_desactives": _count("SIM", STATUTS_DESACTIVES),
            "sim_attente": _count("SIM", STATUTS_ATTENTE),
            "suspendus": tous_suspendus,
            "contrats": all_c[:5],
            "a_suspendu": tous_suspendus > 0,
        }
    except Exception as e:
        logger.error("_fetch_contract_status error: %s", e)
        return {
            "total": 0, "suspendus": 0,
            "esim_actives": 0, "esim_suspendus": 0, "esim_desactives": 0, "esim_attente": 0,
            "sim_actives": 0, "sim_suspendus": 0, "sim_desactives": 0, "sim_attente": 0,
            "contrats": [], "a_suspendu": False,
        }
    finally:
        conn.close()


def _fetch_previous_plans(client_id: int) -> List[dict]:
    conn = get_db_connection()
    try:
        rows = conn.execute(text("""
            SELECT id, plan_data, summary, status, created_at
            FROM payment_plans WHERE client_id = :cid
            ORDER BY created_at DESC LIMIT 3
        """), {"cid": client_id}).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _build_rich_client_context(client_id: int) -> Tuple[str, float, dict]:
    client = _fetch_client_info(client_id)
    scoring = calculer_score_client(client_id)
    invoices = _fetch_unpaid_invoices(client_id)
    pay_history = _fetch_payment_history(client_id, months=6)
    contracts = _fetch_contract_status(client_id)
    previous_plans = _fetch_previous_plans(client_id)
    total = sum(float(inv["montant"]) for inv in invoices)

    if not client:
        return "Client not found in the system.", 0.0, scoring

    nb_paie_temps = sum(1 for p in pay_history if p["statut_paiement"] == "A_TEMPS")
    if pay_history:
        taux_ponctualite = round(nb_paie_temps / len(pay_history) * 100)
        tendance = f"{taux_ponctualite}% on time over the last {len(pay_history)} invoices"
    else:
        tendance = "New client or no recent history"

    ctx = "=== CLIENT PROFILE ===\n"
    ctx += f"Name: {client['prenom']} {client['nom']}\n"
    ctx += f"Contact: {client.get('email_or_phone', 'N/A')}\n\n"

    ctx += "=== SCORE & STRATEGY ===\n"
    ctx += f"Score: {scoring['score']}/100 — {scoring['categorie']}\n"
    ctx += f"Tolerance: {scoring['tolerance_jours']} day(s)\n"
    ctx += f"Strategy: {scoring['strategie']}\n"
    ctx += f"Purchase blocked: {'YES' if scoring['blocage_achat'] else 'no'}\n\n"

    ctx += "=== CURRENT FINANCIAL SITUATION ===\n"
    ctx += f"Unpaid invoices: {len(invoices)} | Total owed: {total:.2f} {CURRENCY_SYMBOL}\n"
    for inv in invoices:
        ctx += f"  - Invoice #{inv['id']}: {float(inv['montant']):.2f} {CURRENCY_SYMBOL} — {inv['jours_retard']}d overdue ({inv['type_ligne']})\n"
    ctx += f"\nPayment history (6 months): {tendance}\n"
    if pay_history:
        nb_show = min(3, len(pay_history))
        ctx += f"Last {nb_show} payments:\n"
        for p in pay_history[:nb_show]:
            ctx += f"  - {float(p['montant']):.2f} {CURRENCY_SYMBOL} — {p['statut_paiement']} ({p['date_paiement']})\n"

    ctx += "\n=== LINE STATUS ===\n"
    ctx += f"eSIM -> Active: {contracts['esim_actives']} | Suspended: {contracts['esim_suspendus']} | Deactivated: {contracts['esim_desactives']} | Pending: {contracts['esim_attente']}\n"
    ctx += f"SIM  -> Active: {contracts['sim_actives']} | Suspended: {contracts['sim_suspendus']} | Deactivated: {contracts['sim_desactives']} | Pending: {contracts['sim_attente']}\n"
    if contracts["a_suspendu"]:
        ctx += "NOTE: at least one line is currently suspended.\n"

    if previous_plans:
        ctx += "\n=== PREVIOUS PAYMENT PLANS ===\n"
        for pl in previous_plans:
            ctx += f"  - Plan from {str(pl['created_at'])[:10]} — Status: {pl['status']}\n"

    stats = scoring.get("stats", {})
    ctx += "\n=== DETAILED SCORING DATA ===\n"
    ctx += (
        f"Total invoices: {stats.get('total_factures', 0)} | "
        f"On time: {stats.get('payees_a_temps', 0)} | "
        f"Late: {stats.get('en_retard', 0)} | "
        f"Unpaid: {stats.get('impayees', 0)}\n"
    )

    return ctx, total, scoring


def _get_rag_context(message: str, client_context: str) -> Tuple[str, List[str]]:
    retriever = get_retriever()

    if not retriever.is_ready:
        # Embedded fallback knowledge base — see README for why this exists:
        # the agent stays useful even before `python -m assistant.rag.indexer` has run.
        fallback = f"""EMBEDDED {COMPANY_NAME.upper()} KNOWLEDGE BASE:

Collections scoring: clients are scored 0-100. Score >= 85 (EXCELLENT):
up to 4 installments, no upfront payment. Score 65-84 (RELIABLE): up to 3
installments. Score 45-64 (AVERAGE): 2 installments, 20% upfront. Score
25-44 (RISK): 2 installments within 7 days, 30% upfront. Score < 25
(CRITICAL): full payment within 48h.

Collections process: overdue invoice -> reminder -> tolerance exceeded ->
line suspended -> 7-day grace period -> unpaid -> permanent deactivation.
Reactivation is automatic once payment is received."""
        return fallback, ["knowledge_base_embedded"]

    enriched_query = f"{message} client score {client_context[:200]}"
    rag_context = retriever.retrieve_as_context(enriched_query, top_k=5)
    rag_sources = [r.source for r in retriever.retrieve(message, top_k=3)]
    return rag_context, rag_sources


def _build_welcome(prenom: str, scoring: dict, total: float, invoices: List[dict],
                    contracts: dict, previous_plans: List[dict]) -> Tuple[str, List[dict]]:
    score, categorie, couleur = scoring["score"], scoring["categorie"], scoring["couleur"]

    if contracts["a_suspendu"] and total > 0:
        msg_ctx = (
            f"Your line is currently **suspended** and you have **{total:.2f} {CURRENCY_SYMBOL}** "
            "to settle. I can offer you a payment plan right away."
        )
        options = [
            {"id": "w2", "label": "\U0001F4B3 See my payment options", "message": "I want to see my payment options", "type": "quick_reply"},
            {"id": "w1", "label": "\U0001F4CB My unpaid invoices", "message": "Show my unpaid invoices in detail", "type": "quick_reply"},
        ]
    elif total > 0:
        msg_ctx = (
            f"You have **{len(invoices)} invoice(s)** pending for a total of **{total:.2f} {CURRENCY_SYMBOL}**. "
            f"Your {couleur} score qualifies you for a tailored plan."
        )
        options = [
            {"id": "w2", "label": "\U0001F4B3 Payment options", "message": "I want to see my payment options", "type": "quick_reply"},
            {"id": "w1", "label": "\U0001F4CB My invoices", "message": "Show my unpaid invoices in detail", "type": "quick_reply"},
            {"id": "w3", "label": "\U0001F4DE Talk to an advisor", "message": f"How do I contact a {COMPANY_NAME} advisor?", "type": "quick_reply"},
        ]
    elif previous_plans:
        msg_ctx = "You're all caught up. Your last payment plan was recorded successfully."
        options = [
            {"id": "w1", "label": "\U0001F4CB My history", "message": "Show my billing history", "type": "quick_reply"},
            {"id": "w3", "label": "\U0001F4DE Contact us", "message": f"How do I contact {COMPANY_NAME}?", "type": "quick_reply"},
        ]
    else:
        msg_ctx = "Your account is in good standing. I'm here if you have any questions."
        options = [
            {"id": "w1", "label": "\U0001F4CB My invoices", "message": "Show my billing history", "type": "quick_reply"},
            {"id": "w3", "label": "\U0001F4DE Contact us", "message": f"How do I contact {COMPANY_NAME}?", "type": "quick_reply"},
        ]

    welcome = WELCOME_TEMPLATE.format(prenom=prenom, score=score, categorie=categorie, message_contextuel=msg_ctx)
    return welcome, options


class AssistantAgent:

    def auto_session(self, client_id: int) -> dict:
        existing = session_memory.find_active_session(client_id)
        session_id = existing["session_id"] if existing else session_memory.create_session(client_id)

        client = _fetch_client_info(client_id)
        scoring = calculer_score_client(client_id)
        invoices = _fetch_unpaid_invoices(client_id)
        contracts = _fetch_contract_status(client_id)
        plans = _fetch_previous_plans(client_id)
        total = sum(float(inv["montant"]) for inv in invoices)
        prenom = client["prenom"] if client else "there"

        welcome, options = _build_welcome(prenom, scoring, total, invoices, contracts, plans)
        session_memory.add_message(session_id, "assistant", welcome)

        logger.info("Auto-session | client=%d | score=%d | owed=%.2f | suspended=%s | session=%s",
                    client_id, scoring["score"], total, contracts["a_suspendu"], session_id)
        return {
            "session_id": session_id,
            "response": welcome,
            "options": options,
            "client": {
                "prenom": prenom, "score": scoring["score"], "categorie": scoring["categorie"],
                "couleur": scoring["couleur"], "impaye": round(total, 2),
                "suspendu": contracts["a_suspendu"], "nb_factures_impayees": len(invoices),
            },
        }

    def start_session(self, client_id: int) -> dict:
        return self.auto_session(client_id)

    def _handle_support_logic(self, session_id: str, client_id: int, prenom: str, message: str) -> Optional[dict]:
        neg = get_negotiator()
        detected, complaint_type = neg.detect_support_need(message)
        if not detected:
            return None

        session_memory.increment_complaint_count(session_id)
        complaint_count = session_memory.get_complaint_count(session_id)
        escalate, ctype = should_escalate(message=message, session_complaint_count=complaint_count, triggered_by_system=False)
        ctype = ctype or ComplaintType.GENERAL_COMPLAINT

        if not escalate:
            result = build_first_mention_response(prenom, ctype)
            session_memory.add_message(session_id, "assistant", result["response"])
            return {"session_id": session_id, "response": result["response"], "options": result["options"],
                    "intent": "SUPPORT_SOFT", "error": False}

        result = build_proposal_response(prenom=prenom, complaint_type=ctype, session_id=session_id)
        session_memory.add_message(session_id, "assistant", result["response"])
        return {"session_id": session_id, "response": result["response"], "options": result["options"],
                "intent": "SUPPORT_ESCALATED", "error": False}

    def process_message(self, session_id: str, client_id: int, raw_message: str) -> dict:
        neg = get_negotiator()

        message = sanitize_user_input(raw_message)
        if not message:
            return self._error_response(session_id, "Empty message.")

        if not self._require_owned_session(session_id, client_id):
            logger.info("Session %s expired — recreating.", session_id)
            session_id = session_memory.create_session(client_id)

        client_ctx, total_impaye, scoring = _build_rich_client_context(client_id)
        lt_history = get_client_long_term_history(client_id, limit=5)

        client = _fetch_client_info(client_id)
        prenom = client["prenom"] if client else "there"

        support_response = self._handle_support_logic(session_id, client_id, prenom, message)
        if support_response:
            return support_response

        intent = neg.classify_intent(message)
        # Never log message text: it's personal data (see privacy.py).
        logger.info("Intent=%s | client=%d | msg_len=%d", intent, client_id, len(message))

        rag_context, rag_sources = _get_rag_context(message, client_ctx)

        already_confirmed = bool(session_memory.get_confirmed_plan(session_id))
        plans_actifs = get_client_plans(client_id)
        has_active_plan = any(p.get("status") == "CONFIRMED" for p in plans_actifs)

        auto_show = (
            total_impaye > 0 and not already_confirmed and not has_active_plan
            and intent not in (INTENT_OFF_TOPIC, INTENT_GREETING)
        )
        show_options = auto_show or (
            neg.should_show_options(intent, total_impaye, already_confirmed)
            and not has_active_plan and intent != INTENT_SUPPORT_REQUEST
        )

        neg_ctx = neg.negotiation_context(intent=intent, score=scoring["score"], categorie=scoring["categorie"], total=total_impaye) if show_options else ""
        full_rag = (rag_context + "\n\n" + neg_ctx).strip() if neg_ctx else rag_context
        system = PromptTemplates.system(client_context=client_ctx, rag_context=full_rag, long_term_history=lt_history)

        st_history = session_memory.get_llm_history(session_id)

        try:
            llm = get_groq_client()
            raw_resp = llm.complete_with_system(system_prompt=system, user_message=message, history=st_history, temperature=0.3)
        except Exception as exc:
            # The LLM only ever *narrates*; scoring and the payment planner are
            # pure Python. So when it's unavailable we still serve a complete,
            # correct answer — options included — with canned wording.
            logger.warning("LLM unavailable — serving deterministic reply: %s", exc)
            metrics.DEGRADED_REPLIES.inc()
            fallback_options = (
                neg.get_payment_options(total_impaye, scoring["score"]) if show_options else []
            )
            if fallback_options:
                session_memory.set_proposed_plans(session_id, fallback_options)

            nb_invoices = len(_fetch_unpaid_invoices(client_id))
            response = offline_reply(
                prenom=prenom, total=total_impaye, nb_invoices=nb_invoices,
                scoring=scoring, options=fallback_options,
            )
            session_memory.add_message(session_id, "user", message)
            session_memory.add_message(session_id, "assistant", response)
            return {
                "session_id": session_id, "response": response, "options": fallback_options,
                "intent": intent, "requires_confirmation": len(fallback_options) > 0,
                "rag_sources": rag_sources, "degraded": True, "error": False,
            }

        valid, response = validate_and_sanitize(raw_resp)

        session_memory.add_message(session_id, "user", message)
        session_memory.add_message(session_id, "assistant", response)

        payment_options: List[dict] = []
        if show_options:
            payment_options = neg.get_payment_options(total_impaye, scoring["score"])
            session_memory.set_proposed_plans(session_id, payment_options)

        return {
            "session_id": session_id, "response": response, "options": payment_options,
            "intent": intent, "requires_confirmation": len(payment_options) > 0,
            "rag_sources": rag_sources, "error": False,
        }

    def handle_option_click(self, session_id: str, client_id: int, option_id: str) -> dict:
        self._require_owned_session(session_id, client_id)
        welcome_map = {
            "w1": "Show my unpaid invoices in detail",
            "w2": "I want to see my payment options",
            "w3": f"How do I contact a {COMPANY_NAME} advisor?",
        }
        if option_id in welcome_map:
            return self.process_message(session_id, client_id, welcome_map[option_id])

        try:
            oid = int(option_id)
        except ValueError:
            return self._error_response(session_id, "Invalid option.")

        plans = session_memory.get_proposed_plans(session_id)
        chosen = next((p for p in plans if p["id"] == oid), None)
        if not chosen:
            return self._error_response(session_id, "Option not found.")

        versements = chosen.get("versements", [])
        v_detail = " | ".join(f"{v['label']}: {v['montant']:.2f} {CURRENCY_SYMBOL} on {v['date']}" for v in versements)
        try:
            llm = get_groq_client()
            client_ctx, _, scoring = _build_rich_client_context(client_id)
            system = PromptTemplates.system(client_context=client_ctx, rag_context="", long_term_history="")
            prompt = PromptTemplates.option_clicked(
                option_label=chosen["label"], total=float(chosen["total"]),
                versements_detail=v_detail, advance=float(chosen.get("advance", 0)),
            )
            raw_r = llm.complete_with_system(system_prompt=system, user_message=prompt, temperature=0.3)
            _, response = validate_and_sanitize(raw_r)
        except Exception:
            response = (
                f"You selected: **{chosen['label']}**\n"
                f"Total: **{chosen['total']:.2f} {CURRENCY_SYMBOL}**\n"
                f"Installments: {v_detail}\n\nWould you like to confirm this plan?"
            )

        session_memory.add_message(session_id, "assistant", response)
        session_memory.set_context(session_id, "pending_plan", chosen)

        return {
            "session_id": session_id, "response": response, "options": [],
            "pending_plan": chosen, "requires_confirmation": True,
            "intent": "PLAN_SELECTED", "error": False,
        }

    def confirm_plan(self, session_id: str, client_id: int, option_id: int) -> dict:
        if not self._require_owned_session(session_id, client_id):
            return {"success": False, "message": SESSION_EXPIRED_MSG}

        active = get_active_plan(client_id)
        if active:
            return self._already_confirmed(active["id"])

        plans = session_memory.get_proposed_plans(session_id)
        chosen = next((p for p in plans if p["id"] == option_id), None) or session_memory.get_context(session_id, "pending_plan")
        if not chosen:
            return {"success": False, "message": "Invalid option."}

        # Claim before the slow part (LLM summary) so a double click or a
        # retried request can't confirm the same plan twice.
        if not session_memory.claim_confirmation(session_id, chosen):
            return self._already_confirmed(None)

        confirmed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        neg = get_negotiator()
        summary = neg.generate_plan_summary(client_id, chosen, confirmed_at)

        try:
            plan_id = save_payment_plan(client_id, session_id, chosen, summary)
        except PlanAlreadyConfirmedError:
            # Another session confirmed a plan for this client in the meantime.
            active = get_active_plan(client_id)
            return self._already_confirmed(active["id"] if active else None)
        if plan_id is None:
            session_memory.release_confirmation(session_id)
            return {"success": False, "message": "We couldn't save your plan just now. Please try again in a moment."}

        save_interaction_summary(client_id, session_id, summary)
        notify_admin(client_id, plan_id, summary)
        metrics.PLANS_CONFIRMED.inc()

        client = _fetch_client_info(client_id)
        prenom = client["prenom"] if client else ""
        confirm_msg = (
            f"**Plan confirmed! Thank you, {prenom}.**\n\n{summary}\n\n"
            "A confirmation summary will be sent to you by email shortly. "
            "Our collections team has been automatically notified.\n\n"
            "Is there anything else I can help you with?"
        )
        session_memory.add_message(session_id, "assistant", confirm_msg)
        logger.info("Plan #%s confirmed | session=%s | client=%d", plan_id, session_id, client_id)
        return {"success": True, "plan_id": plan_id, "plan": chosen, "summary": summary, "message": confirm_msg}

    def create_support_ticket(self, session_id: str, client_id: int, complaint_type: str) -> dict:
        self._require_owned_session(session_id, client_id)
        client = _fetch_client_info(client_id)
        if not client:
            return {"success": False, "message": "Client not found."}

        result = auto_create__ticket(client=client, complaint_type=ComplaintType(complaint_type))
        if result.get("response"):
            session_memory.add_message(session_id, "assistant", result["response"])
        return result

    def _already_confirmed(self, plan_id: Optional[int]) -> dict:
        return {
            "success": False, "already_confirmed": True, "plan_id": plan_id,
            "message": "You already have a confirmed payment plan. "
                       f"Contact {COMPANY_NAME} support if you need to change it.",
        }

    def _require_owned_session(self, session_id: str, client_id: int) -> bool:
        """True if the session is live and belongs to client_id, False if it
        doesn't exist (never created or expired). Raises SessionOwnershipError
        if it belongs to another client — session ids arrive in request
        bodies, so they must never be trusted on their own."""
        owner = session_memory.owner_of(session_id)
        if owner is None:
            return False
        if owner != client_id:
            logger.warning("Client %d tried to use session %s owned by client %d", client_id, session_id, owner)
            raise SessionOwnershipError(session_id)
        session_memory.get_session(session_id)  # mark active
        return True

    def _error_response(self, session_id: str, msg: str) -> dict:
        return {"session_id": session_id, "response": msg, "options": [], "error": True}


_agent: Optional[AssistantAgent] = None


def get_agent() -> AssistantAgent:
    global _agent
    if _agent is None:
        _agent = AssistantAgent()
    return _agent
