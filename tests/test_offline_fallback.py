"""
tests/test_offline_fallback.py — The agent must stay useful with no LLM.

The README promises that a clone with no GROQ_API_KEY still gets working
scoring and payment plans, and that only the free-text wording degrades.
These tests pin that promise down: the money math is pure Python, so the
offline path has to surface the *same* amounts the planner computed.
"""
import pytest

from assistant.llm.prompt_templates import offline_reply
from assistant.llm.groq_client import GroqLLMClient
from assistant.tools.payment_planner import build_payment_options


SCORING_RISK = {"score": 34, "categorie": "RISQUE"}


class TestOfflineReply:

    def test_reports_score_and_balance(self):
        msg = offline_reply("Karim", 147.04, 2, SCORING_RISK, options=[])
        assert "Karim" in msg
        assert "34/100" in msg
        assert "RISQUE" in msg
        assert "147.04" in msg

    def test_zero_balance_says_account_is_clear(self):
        msg = offline_reply("Karim", 0.0, 0, SCORING_RISK, options=[])
        assert "up to date" in msg
        assert "Payment plans available" not in msg

    def test_every_installment_is_listed(self):
        options = build_payment_options(147.04, 34)
        msg = offline_reply("Karim", 147.04, 2, SCORING_RISK, options=options)

        assert "Payment plans available" in msg
        for opt in options:
            assert opt["label"] in msg
            for v in opt["versements"]:
                # The amount the planner computed must appear verbatim —
                # the offline path must never round or restate it.
                assert f"{v['montant']:.2f}" in msg
                assert v["date"] in msg

    def test_no_options_means_no_plan_section(self):
        msg = offline_reply("Karim", 147.04, 2, SCORING_RISK, options=None)
        assert "Payment plans available" not in msg


class TestUnconfiguredClientShortCircuits:

    def test_complete_raises_without_touching_the_network(self, monkeypatch):
        client = GroqLLMClient()
        monkeypatch.setattr(client, "is_configured", False)

        # If this ever performs a real request it will hang or 401 instead of
        # raising immediately — that regression is what this guards against.
        def _explode(*args, **kwargs):
            raise AssertionError("network call attempted with no API key")

        monkeypatch.setattr(client._client.chat.completions, "create", _explode)

        with pytest.raises(RuntimeError, match="not configured"):
            client.complete([{"role": "user", "content": "hi"}])
