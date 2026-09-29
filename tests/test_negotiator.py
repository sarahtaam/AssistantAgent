"""
Tests for the local-fast-path intent classification in negotiator.py.
These never touch the LLM, so they're fast, free, and deterministic —
exactly what you want covering the code path that handles the majority
of real traffic.
"""
from assistant.agent.negotiator import (
    Negotiator, INTENT_PLAN_REQUEST, INTENT_INVOICE_INFO, INTENT_COMPLAINT,
    INTENT_REACTIVATION, INTENT_GREETING, INTENT_CONFIRM, INTENT_SUPPORT_REQUEST,
)

neg = Negotiator()


class TestIntentClassification:
    def test_greeting(self):
        assert neg.classify_intent("Bonjour") == INTENT_GREETING

    def test_plan_request(self):
        assert neg.classify_intent("Je veux échelonner mon paiement") == INTENT_PLAN_REQUEST

    def test_invoice_info(self):
        assert neg.classify_intent("Combien je dois sur ma facture ?") == INTENT_INVOICE_INFO

    def test_complaint(self):
        assert neg.classify_intent("Cette facture est incorrecte, je conteste") == INTENT_COMPLAINT

    def test_reactivation(self):
        assert neg.classify_intent("Ma ligne est suspendue, comment la réactiver ?") == INTENT_REACTIVATION

    def test_confirm(self):
        assert neg.classify_intent("Oui d'accord, je confirme") == INTENT_CONFIRM

    def test_confirm_numeric_option(self):
        assert neg.classify_intent("2") == INTENT_CONFIRM

    def test_support_request(self):
        assert neg.classify_intent("mon paiement a échoué, bug application") == INTENT_SUPPORT_REQUEST


class TestSupportDetection:
    def test_face_verification_failure_detected(self):
        detected, ctype = neg.detect_support_need("ma vérification biométrie a échoué")
        assert detected
        assert ctype.value == "FACE_VERIFY_FAILED"

    def test_payment_failure_detected(self):
        detected, ctype = neg.detect_support_need("mon paiement a échoué")
        assert detected
        assert ctype.value == "PAYMENT_FAILED"

    def test_neutral_message_not_flagged(self):
        detected, _ = neg.detect_support_need("Bonjour, comment allez-vous ?")
        assert not detected


class TestPaymentOptionsGating:
    def test_should_show_options_for_plan_request(self):
        assert neg.should_show_options(INTENT_PLAN_REQUEST, total_amount=100.0)

    def test_should_not_show_options_with_zero_balance(self):
        assert not neg.should_show_options(INTENT_PLAN_REQUEST, total_amount=0)

    def test_should_not_show_options_if_already_confirmed(self):
        assert not neg.should_show_options(INTENT_PLAN_REQUEST, total_amount=100.0, session_has_confirmed=True)
