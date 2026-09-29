"""
Tests for assistant/tools/payment_planner.py — deterministic financial
logic, which is exactly the kind of code that should never ship untested,
since it's the one place a bug means telling a real client the wrong
amount.
"""
import pytest

from assistant.tools.payment_planner import (
    build_payment_options,
    generate_installment_schedule,
    validate_plan,
)


class TestInstallmentSchedule:
    def test_schedule_sums_to_total(self):
        schedule = generate_installment_schedule(total_amount=300.0, nb_installments=3, interval_days=30)
        assert round(sum(v["montant"] for v in schedule), 2) == 300.0

    def test_schedule_with_advance_sums_to_total(self):
        schedule = generate_installment_schedule(
            total_amount=250.0, nb_installments=2, interval_days=7, advance_pct=0.30
        )
        assert round(sum(v["montant"] for v in schedule), 2) == 250.0
        assert schedule[0]["label"] == "Upfront payment"

    def test_rounding_remainder_absorbed_by_last_installment(self):
        # 100 / 3 doesn't divide evenly — make sure cents aren't lost or invented
        schedule = generate_installment_schedule(total_amount=100.0, nb_installments=3, interval_days=30)
        assert round(sum(v["montant"] for v in schedule), 2) == 100.0

    def test_zero_amount_returns_empty_schedule(self):
        assert generate_installment_schedule(total_amount=0, nb_installments=3, interval_days=30) == []

    def test_negative_amount_returns_empty_schedule(self):
        assert generate_installment_schedule(total_amount=-50, nb_installments=3, interval_days=30) == []


class TestPaymentOptions:
    @pytest.mark.parametrize("score,expected_max_installments", [
        (95, 4), (70, 3), (50, 2), (30, 2), (10, 1),
    ])
    def test_max_installments_matches_score_tier(self, score, expected_max_installments):
        options = build_payment_options(total=200.0, score=score)
        assert max(o["nb"] for o in options) == expected_max_installments

    def test_options_sorted_most_spread_to_least(self):
        options = build_payment_options(total=200.0, score=95)
        nb_values = [o["nb"] for o in options]
        assert nb_values == sorted(nb_values, reverse=True)

    def test_zero_total_returns_no_options(self):
        assert build_payment_options(total=0, score=80) == []

    def test_every_option_schedule_sums_to_total(self):
        for score in (95, 70, 50, 30, 10):
            for option in build_payment_options(total=473.50, score=score):
                assert round(sum(v["montant"] for v in option["versements"]), 2) == 473.50


class TestValidatePlan:
    def test_plan_within_tier_limit_is_valid(self):
        plan = {"nb": 2, "versements": [{"numero": 1}, {"numero": 2}]}
        valid, _ = validate_plan(plan, score=95)
        assert valid

    def test_plan_exceeding_tier_limit_is_invalid(self):
        # score 10 only allows 1 installment
        plan = {"nb": 3, "versements": [{"numero": 1}, {"numero": 2}, {"numero": 3}]}
        valid, msg = validate_plan(plan, score=10)
        assert not valid
        assert "exceeds" in msg
