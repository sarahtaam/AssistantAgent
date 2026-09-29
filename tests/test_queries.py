"""
tests/test_queries.py — The data-access queries against a real database.

Runs on SQLite locally and on Postgres in CI (DATABASE_URL), which is what
proves the queries no longer depend on SQLite-only functions.
"""
from datetime import date

import pytest

from assistant.agent.orchestrator import (
    _fetch_contract_status, _fetch_payment_history, _fetch_unpaid_invoices,
)
from db import to_date
from recouvrement.predict import FEATURES, extraire_features_client
from recouvrement.scoring import calculer_score_client

# seed.py profiles: client 1 = "risk" (2 unpaid), 5 = "critical" (4 unpaid), 4 = "excellent" (0 unpaid)
RISK_CLIENT, CRITICAL_CLIENT, EXCELLENT_CLIENT = 1, 5, 4


@pytest.fixture(autouse=True)
def _db(seeded_db):
    yield


def test_unpaid_invoices_have_days_overdue():
    invoices = _fetch_unpaid_invoices(CRITICAL_CLIENT)
    assert len(invoices) == 4
    for inv in invoices:
        # seed.py puts unpaid due dates 5-90 days in the past
        assert 5 <= inv["jours_retard"] <= 90
        assert inv["jours_retard"] == (date.today() - to_date(inv["date_echeance"])).days
        assert inv["type_ligne"] == "ESIM"
        assert float(inv["montant"]) > 0


def test_excellent_client_has_no_unpaid_invoices():
    assert _fetch_unpaid_invoices(EXCELLENT_CLIENT) == []


def test_payment_history_is_limited_to_recent_months():
    history = _fetch_payment_history(CRITICAL_CLIENT, months=6)
    assert history, "seed.py creates at least 3 paid invoices per client"
    cutoff = date(date.today().year - 1, date.today().month, 1)
    for p in history:
        assert to_date(p["date_paiement"]) >= cutoff
        assert p["statut_paiement"] in ("A_TEMPS", "EN_RETARD")
        assert (p["jours_retard"] > 0) == (p["statut_paiement"] == "EN_RETARD")


def test_payment_history_window_excludes_older_payments():
    assert _fetch_payment_history(CRITICAL_CLIENT, months=0) == []


def test_contract_status():
    status = _fetch_contract_status(RISK_CLIENT)
    assert status["total"] == 1


def test_rules_score_is_in_range():
    for client_id in (RISK_CLIENT, CRITICAL_CLIENT, EXCELLENT_CLIENT):
        score = calculer_score_client(client_id)
        assert 0 <= score["score"] <= 100
    assert calculer_score_client(EXCELLENT_CLIENT)["score"] > calculer_score_client(CRITICAL_CLIENT)["score"]


def test_ml_features_extract_on_this_database():
    features = extraire_features_client(CRITICAL_CLIENT)
    assert set(features) == set(FEATURES)
    assert features["nb_comptes"] == 1
    assert features["dette_impayee"] > 0
    assert features["anciennete_jours"] >= 0
