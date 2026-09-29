"""
assistant/tools/payment_planner.py — Installment plan generation and validation.

Design decision (kept from the original): the LLM never computes amounts.
All money math lives here, in deterministic, unit-testable code. The LLM
only explains a plan that this module already calculated. This is the
single most important guardrail in the whole system — an LLM hallucinating
a payment amount is a much worse failure mode than a wrong chatbot answer.
"""
import math
from datetime import datetime, timedelta
from typing import List, Optional

from config import SCORE_TIERS, CURRENCY_SYMBOL


def _get_constraints(score: int) -> dict:
    for (lo, hi), (max_inst, interval_days, advance_pct, _label, _color) in SCORE_TIERS.items():
        if lo <= score <= hi:
            return {"max_installments": max_inst, "interval_days": interval_days, "advance_pct": advance_pct}
    return {"max_installments": 1, "interval_days": 2, "advance_pct": 1.00}


def generate_installment_schedule(
    total_amount: float,
    nb_installments: int,
    interval_days: int,
    advance_pct: float = 0.0,
    start_date: Optional[datetime] = None,
) -> List[dict]:
    """Computes a precise schedule with an optional upfront installment.
    Rounding remainder is absorbed into the final installment."""
    if total_amount <= 0:
        return []

    start = start_date or datetime.now()
    schedule: List[dict] = []

    advance = round(total_amount * advance_pct, 2)
    remaining = total_amount - advance

    if advance > 0:
        schedule.append({
            "numero": 0,
            "label": "Upfront payment",
            "montant": advance,
            "date": start.strftime("%Y-%m-%d"),
        })

    base = math.floor(remaining / nb_installments * 100) / 100
    last_adj = round(remaining - base * nb_installments, 2)

    for i in range(nb_installments):
        due_date = start + timedelta(days=interval_days * (i + 1))
        amount = base if i < nb_installments - 1 else base + last_adj
        schedule.append({
            "numero": i + 1,
            "label": f"Installment {i + 1}/{nb_installments}",
            "montant": round(amount, 2),
            "date": due_date.strftime("%Y-%m-%d"),
        })

    return schedule


def _score_emoji(score: int) -> str:
    if score >= 85:
        return "\U0001F7E2"
    if score >= 65:
        return "\U0001F535"
    if score >= 45:
        return "\U0001F7E1"
    if score >= 25:
        return "\U0001F7E0"
    return "\U0001F534"


def build_payment_options(total: float, score: int) -> List[dict]:
    """Builds the full set of available payment options for this client,
    sorted from most spread-out to least."""
    if total <= 0:
        return []

    constraints = _get_constraints(score)
    max_inst = constraints["max_installments"]
    interval = constraints["interval_days"]
    advance_pct = constraints["advance_pct"]

    options: List[dict] = []
    option_id = 1

    for nb in range(max_inst, 0, -1):
        schedule = generate_installment_schedule(
            total_amount=total,
            nb_installments=nb,
            interval_days=interval,
            advance_pct=advance_pct if nb > 1 else 0,
        )
        advance_amount = round(total * advance_pct, 2) if nb > 1 else 0

        if advance_pct >= 1.0 or (nb == 1 and nb == max_inst):
            label = f"Pay in full within {interval} day(s)"
            emoji = "\U0001F4B3"
        elif interval >= 30:
            label = f"{nb}-month plan"
            emoji = _score_emoji(score)
        else:
            label = f"{nb}-installment plan / {nb * interval} days"
            emoji = _score_emoji(score)

        options.append({
            "id": option_id,
            "label": label,
            "emoji": emoji,
            "total": round(total, 2),
            "advance": advance_amount,
            "versements": schedule,
            "nb": nb,
            "interval": interval,
        })
        option_id += 1

    return options


def validate_plan(plan: dict, score: int) -> tuple[bool, str]:
    """Verifies a plan respects the scoring-tier constraints."""
    constraints = _get_constraints(score)
    nb = plan.get("nb", len([v for v in plan.get("versements", []) if v["numero"] > 0]))

    if nb > constraints["max_installments"]:
        return False, (
            f"Number of installments ({nb}) exceeds the maximum allowed "
            f"for this risk profile ({constraints['max_installments']})."
        )
    return True, "Valid plan."


def format_schedule_for_prompt(schedule: List[dict]) -> str:
    """Human-readable format for injecting into an LLM prompt."""
    lines = []
    for v in schedule:
        lines.append(f"  {v['label']}: {CURRENCY_SYMBOL}{v['montant']:.2f} — {v['date']}")
    return "\n".join(lines)
