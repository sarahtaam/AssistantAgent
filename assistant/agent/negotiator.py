"""
assistant/agent/negotiator.py — Intent classification and negotiation engine.

Two-layer intent classification: a fast local keyword layer handles the
large majority of messages for free with no LLM round-trip; only genuinely
ambiguous messages fall through to an LLM classification call. This keeps
both latency and per-conversation LLM cost down.

Note on language: the keyword lists are French because this demo targets
a French-speaking market (matching the original deployment context).
Swap `_DOMAIN_WORDS` / `_PLAN_KEYWORDS` / etc. for another language to
retarget — the classification logic itself is language-agnostic.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from assistant.llm.groq_client import get_groq_client
from assistant.llm.prompt_templates import PromptTemplates
from assistant.tools.payment_planner import build_payment_options
from assistant.support_service import ComplaintType

logger = logging.getLogger(__name__)

# ── Intent constants ────────────────────────────────────────────────────
INTENT_PLAN_REQUEST = "PLAN_REQUEST"
INTENT_INVOICE_INFO = "INVOICE_INFO"
INTENT_COMPLAINT = "COMPLAINT"
INTENT_REACTIVATION = "REACTIVATION"
INTENT_GREETING = "GREETING"
INTENT_CONFIRM = "CONFIRM"
INTENT_OFF_TOPIC = "OFF_TOPIC"
INTENT_SUPPORT_REQUEST = "SUPPORT_REQUEST"
INTENT_TECHNICAL_ISSUE = "TECHNICAL_ISSUE"
INTENT_OTHER = "OTHER"

_VALID_INTENTS = frozenset([
    INTENT_PLAN_REQUEST, INTENT_INVOICE_INFO, INTENT_COMPLAINT,
    INTENT_REACTIVATION, INTENT_GREETING, INTENT_CONFIRM, INTENT_SUPPORT_REQUEST,
    INTENT_TECHNICAL_ISSUE, INTENT_OFF_TOPIC, INTENT_OTHER,
])
PAYMENT_INTENTS = frozenset([INTENT_PLAN_REQUEST, INTENT_COMPLAINT, INTENT_SUPPORT_REQUEST, INTENT_OTHER])

# ── Keyword sets — local fast-path classification ──────────────────────
# If any of these appear, the message is never OFF_TOPIC.
_DOMAIN_WORDS = frozenset([
    "facture", "factures", "paiement", "paiements", "payer", "payé",
    "impayé", "dette", "solde", "montant", "recouvrement", "règlement",
    "abonnement", "contrat", "ligne", "sim", "esim", "forfait",
    "suspension", "suspendu", "suspendue", "réactiver", "réactivation",
    "débloquer", "résiliation", "score", "service", "plan", "mensualité",
    "versement", "échéance", "acompte", "délai", "facilité", "échelonner",
    "étaler", "rembours", "régulariser", "situation", "compte", "retard",
    "relance", "option", "choisir", "choix", "aide", "aider", "comment",
    "oui", "non", "ok", "d'accord", "confirme", "confirmer", "accepte",
    "je veux", "je souhaite", "je voudrais", "possible",
    "réseau", "service client", "assistance", "rdv", "rendez-vous",
    "boutique", "agence",
])

_PLAN_KEYWORDS = frozenset([
    "plan", "paiement", "payer", "régler", "mensualité", "mensuel",
    "échelonner", "étaler", "versement", "rembours", "délai", "option",
    "facilité", "accord", "proposition", "possible", "modalité",
    "échelonnement", "différé", "acompte", "arrangement", "régulariser",
])

_INVOICE_KEYWORDS = frozenset([
    "facture", "montant", "solde", "dette", "combien", "impayé", "retard",
    "rappel", "relance", "historique", "détail", "montrer facture",
    "état de ma facture", "état de mes factures", "état de la facture",
    "état de l'impayé", "détail de ma facture", "détail de mes factures",
    "score", "mon score", "mon crédit", "crédit", "solde de ma facture",
    "solde de mes factures",
])

_REACTIVATION_KEYWORDS = frozenset([
    "réactiver", "débloquer", "rétablir", "reactivation", "réactiv",
    "suspendu", "suspendue", "suspension", "bloqué", "accès", "renouvellement",
])

_COMPLAINT_KEYWORDS = frozenset([
    "erreur", "contesté", "injuste", "faux", "incorrect", "tort",
    "réclamation", "litige", "plainte", "problème", "contester",
    "injustement", "abusif", "défaut",
])

_SUPPORT_KEYWORDS = frozenset([
    "problème", "bug", "erreur", "support", "assistance",
    "paiement refusé", "paiement échoué", "vérification échouée",
    "caméra", "face id", "biométrie", "ticket", "impossible", "bloqué",
    "application", "connexion", "help", "service client", "aide urgente",
    "urgence", "aide svp", "soutien", "difficulté", "assistance technique",
    "problème technique", "problème de paiement", "problème de connexion",
    "problème de vérification", "problème de caméra", "problème de face id",
    "problème de biométrie", "problème de ticket", "problème d'application",
    "problème de service client", "le paiement ne fonctionne pas",
    "face verification failed", "vérification faciale échouée",
    "biométrie ne fonctionne pas", "bug application",
])

_CONFIRM_KEYWORDS = frozenset([
    "oui", "ok", "d'accord", "confirme", "confirmer", "accepte",
    "je prends", "je choisis", "je valide", "validé", "parfait",
    "entendu", "bien sûr", "option 1", "option 2", "option 3",
    "plan 1", "plan 2", "plan 3", "versement 1", "versement 2", "versement 3",
    "1", "2", "3",
])

_GREETING_KEYWORDS = frozenset([
    "bonjour", "bonsoir", "salut", "hello", "bonne journée", "bjr", "slt",
    "hey", "hi", "coucou", "bienvenue", "slm", "salam", "salam alaykom",
    "salam aleykoum", "merci", "merci d'avoir répondu", "merci de votre réponse",
])


class Negotiator:
    """Negotiation engine. Thread-safe — no mutable state between requests."""

    def classify_intent(self, message: str) -> str:
        """
        Multi-layer classification:
          1. Confirmation / support keywords (highest priority)
          2. Local keyword detection (fast, free, reliable)
          3. LLM fallback, only for genuine ambiguity with no domain keyword hit
        """
        lower = message.lower().strip()

        if any(kw in lower for kw in _CONFIRM_KEYWORDS):
            return INTENT_CONFIRM
        if any(kw in lower for kw in _SUPPORT_KEYWORDS):
            return INTENT_SUPPORT_REQUEST
        if any(kw in lower for kw in _GREETING_KEYWORDS):
            return INTENT_GREETING
        if any(kw in lower for kw in _PLAN_KEYWORDS):
            return INTENT_PLAN_REQUEST
        if any(kw in lower for kw in _REACTIVATION_KEYWORDS):
            return INTENT_REACTIVATION
        if any(kw in lower for kw in _COMPLAINT_KEYWORDS):
            return INTENT_COMPLAINT
        if any(kw in lower for kw in _INVOICE_KEYWORDS):
            return INTENT_INVOICE_INFO
        if any(kw in lower for kw in _DOMAIN_WORDS):
            logger.debug("Domain keyword detected — intent=OTHER (not OFF_TOPIC)")
            return INTENT_OTHER
        if len(lower.split()) <= 3:
            return INTENT_OTHER

        return self._classify_via_llm(message)

    def detect_support_need(self, message: str) -> tuple[bool, ComplaintType]:
        lower = message.lower()
        payment_words = ("paiement", "carte", "payment")
        failure_words = ("échou", "échec", "refus", "failed", "refused")
        if any(p in lower for p in payment_words) and any(f in lower for f in failure_words):
            return True, ComplaintType.PAYMENT_FAILED
        if any(k in lower for k in ["face id", "vérification", "biométrie", "caméra", "selfie"]):
            return True, ComplaintType.FACE_VERIFY_FAILED
        if any(k in lower for k in ["ligne bloquée", "sim bloquée", "esim bloquée", "pas de réseau", "problème technique"]):
            return True, ComplaintType.TECHNICAL_ISSUE
        if any(k in lower for k in _SUPPORT_KEYWORDS):
            return True, ComplaintType.GENERAL_COMPLAINT
        return False, ComplaintType.GENERAL_COMPLAINT

    def _classify_via_llm(self, message: str) -> str:
        try:
            llm = get_groq_client()
            prompt = PromptTemplates.intent_with_examples(message)
            result = llm.complete([{"role": "user", "content": prompt}], max_tokens=15, temperature=0.0)
            intent = result.strip().upper().split()[0]
            if intent not in _VALID_INTENTS:
                return INTENT_OTHER
            return intent
        except Exception as exc:
            logger.warning("LLM classification failed -> OTHER: %s", exc)
            return INTENT_OTHER

    def get_payment_options(self, total_amount: float, score: int) -> List[dict]:
        if total_amount <= 0:
            return []
        options = build_payment_options(total_amount, score)
        logger.info("Options: %d for %.2f (score=%d)", len(options), total_amount, score)
        return options

    def should_show_options(self, intent: str, total_amount: float, session_has_confirmed: bool = False) -> bool:
        if session_has_confirmed or total_amount <= 0:
            return False
        return intent in PAYMENT_INTENTS

    def negotiation_context(self, intent: str, score: int, categorie: str, total: float) -> str:
        if intent not in PAYMENT_INTENTS or total <= 0:
            return ""
        return PromptTemplates.negotiation_context(score=score, categorie=categorie, total=total)

    def generate_plan_summary(self, client_id: int, plan: dict, confirmed_at: str) -> str:
        import json
        try:
            llm = get_groq_client()
            prompt = PromptTemplates.plan_summary(
                client_id=client_id,
                plan_label=plan.get("label", "Payment plan"),
                total=float(plan.get("total", 0)),
                versements=json.dumps(plan.get("versements", []), ensure_ascii=False),
                confirmed_at=confirmed_at,
            )
            return llm.complete([{"role": "user", "content": prompt}], max_tokens=250, temperature=0.1)
        except Exception as exc:
            logger.warning("LLM summary failed: %s", exc)
            v = plan.get("versements", [])
            return (
                f'Plan "{plan.get("label", "?")}" confirmed on {confirmed_at}. '
                f'Total: {plan.get("total", 0):.2f}. {len(v)} installment(s). Status: CONFIRMED.'
            )


_negotiator: Optional[Negotiator] = None


def get_negotiator() -> Negotiator:
    global _negotiator
    if _negotiator is None:
        _negotiator = Negotiator()
    return _negotiator
