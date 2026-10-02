"""
assistant/llm/output_guard.py — Input/output guardrails.

Design principle: domain-relevant responses are never blocked as
off-topic — off-topic detection is the LAST check, applied only to
responses containing zero domain keywords. Real dangers (prompt
injection, data leakage, XSS, hallucinated amounts) are checked first
and always block, regardless of domain relevance.

This ordering fixes a real failure mode from an earlier version: an
over-eager off-topic filter was rejecting legitimate business responses
that happened to phrase things in an unexpected way. Checking domain
relevance first, safety second, eliminated those false positives.
"""
from __future__ import annotations
import re
import logging
from typing import Tuple

import metrics
from config import MAX_REALISTIC_AMOUNT, CURRENCY_SYMBOL

logger = logging.getLogger(__name__)

_DOMAIN_KEYWORDS = frozenset([
    "facture", "paiement", "impayé", "recouvrement", "versement",
    "montant", "dette", "solde", "règlement", "mensualité", "échéance",
    "plan", "acompte", "délai", "régulariser", "abonnement", "contrat",
    "suspension", "réactivation", "résiliation", "ligne", "sim", "esim",
    "forfait", "accès", "score", "client", "compte", "situation", "profil",
    "service", "aider", "assistance", "bonjour", "merci", "cordialement",
    "désolé", "comprendre", "option", "choisir", "confirmer", "validé",
    "retard", "relance", "régularisation", "offre",
    "conseiller", "aide en ligne", "support", "contact", "joindre",
    "appel", "numéro", "téléphone", "service client", "problème technique",
])

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(tes|les|mes|previous|all)\s+(instructions?|règles?|consignes?)", "prompt_injection"),
    (r"oublie\s+(tes|les|toutes?\s+les)\s+(instructions?|règles?)", "prompt_injection"),
    (r"tu\s+es\s+maintenant\s+un\s+autre", "role_override"),
    (r"pretend\s+(?:to\s+be|you\s+are)", "role_override"),
    (r"act\s+as\s+(?:a\s+)?(?:new|different|unrestricted|evil)", "role_override"),
    (r"jailbreak", "jailbreak"),
    (r"DAN\s+mode", "jailbreak"),
    (r"<\s*script", "xss"),
    (r"javascript\s*:", "xss"),
    (r"on(?:click|load|error|mouseover)\s*=", "xss"),
]

_LEAK_PATTERNS: list[tuple[str, str]] = [
    (r"autres?\s+clients?\s+(?:ont|avec|possèdent)", "data_leak"),
    (r"données?\s+personnelles?\s+d[e']un\s+autre", "data_leak"),
    (r"(?:mot\s+de\s+passe|password).{0,20}(?:est|:)", "credentials"),
    (r"api[_\s]?key.{0,20}(?:est|:|sk-)", "credentials"),
    (r"prompt\s+système.*règles?\s+absolues?", "prompt_leak"),
    (r"règles?\s+absolues?.*ne\s+jamais\s+enfreindre", "prompt_leak"),
]

_MAX_RESPONSE_LENGTH = 4_000
_MAX_INPUT_LENGTH = 1_500


def _match_patterns(text: str, patterns) -> Tuple[bool, str]:
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, label
    return False, ""


def _is_on_topic(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _DOMAIN_KEYWORDS)


def _check_amounts(text: str) -> bool:
    """Blocks clearly unrealistic amounts (hallucination guard)."""
    amounts = re.findall(
        r"\b(\d[\d\s]{0,8}(?:[.,]\d{1,2})?)\s*(?:" + re.escape(CURRENCY_SYMBOL) + r"|dollars?|dt|dinars?)",
        text, re.IGNORECASE,
    )
    for raw in amounts:
        try:
            val = float(raw.replace(" ", "").replace(",", "."))
            if val > MAX_REALISTIC_AMOUNT:
                logger.warning("Output guard — unrealistic amount: %.0f", val)
                return False
        except ValueError:
            continue
    return True


def _sanitize_text(text: str) -> str:
    text = re.sub(r"<(?!br|b|i|strong|em)[^>]{1,100}>", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    if len(text) > _MAX_RESPONSE_LENGTH:
        text = text[:_MAX_RESPONSE_LENGTH].rsplit(" ", 1)[0] + " […]"
    return text


def sanitize_user_input(message: str) -> str:
    """Cleans an incoming message. Only blocks real injection attempts —
    preserves every legitimate domain message untouched."""
    if not message:
        return ""
    message = message.strip()[:_MAX_INPUT_LENGTH]

    found, label = _match_patterns(message, _INJECTION_PATTERNS)
    if found:
        # Log the rule that matched, never the text itself (personal data).
        logger.warning("Input guard — %s (msg_len=%d)", label, len(message))
        metrics.GUARDRAIL_BLOCKS.labels(guard="input", reason=label).inc()
        return "I'd like information about my invoices and payment options."
    return message


def validate_and_sanitize(response: str) -> Tuple[bool, str]:
    """
    Validates an LLM response, in priority order:
      0. Domain-relevant content -> skip the off-topic check entirely
      1. Data leakage -> block
      2. Injection echoed in the output -> block
      3. Unrealistic amounts -> block
      4. Genuinely off-topic (zero domain keywords) -> redirect
      5. Final cleanup
    """
    from assistant.llm.prompt_templates import OFF_TOPIC_RESPONSE
    from config import SUPPORT_PHONE

    if not response or not response.strip():
        logger.warning("Output guard — empty response.")
        return False, "I'm having trouble right now. Please try again."

    is_domain = _is_on_topic(response)

    found, label = _match_patterns(response, _LEAK_PATTERNS)
    if found:
        logger.warning("Output guard — leak (%s)", label)
        metrics.GUARDRAIL_BLOCKS.labels(guard="output", reason=label).inc()
        return False, f"I can't display that information. Please contact {SUPPORT_PHONE}."

    found, label = _match_patterns(response, _INJECTION_PATTERNS)
    if found:
        logger.warning("Output guard — output injection (%s)", label)
        metrics.GUARDRAIL_BLOCKS.labels(guard="output", reason=label).inc()
        return False, "I can't respond to that request."

    if not _check_amounts(response):
        metrics.GUARDRAIL_BLOCKS.labels(guard="output", reason="unrealistic_amount").inc()
        return False, f"An inconsistency was detected. Please contact {SUPPORT_PHONE}."

    if not is_domain:
        logger.info("Output guard — response has no domain keywords -> OFF_TOPIC.")
        metrics.GUARDRAIL_BLOCKS.labels(guard="output", reason="off_topic").inc()
        return False, OFF_TOPIC_RESPONSE

    return True, _sanitize_text(response)
