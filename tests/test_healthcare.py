"""Tests for healthcare costs in retirement."""

from __future__ import annotations

import pytest

from monteplan.config.defaults import default_market
from monteplan.config.schema import (
    AccountConfig,
    HealthcareCost,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate


def _terminal_wealth(healthcare: HealthcareCost | None, retirement_age: int = 60) -> float:
    """Terminal wealth of a taxable account without returns, inflation or taxes."""
    market = default_market().model_copy(
        update={
            "expected_annual_returns": [0.0] * 6,
            "annual_volatilities": [0.0] * 6,
            "inflation_mean": 0.0,
            "inflation_vol": 0.0,
        }
    )
    plan = PlanConfig(
        current_age=60,
        retirement_age=retirement_age,
        end_age=62,
        monthly_spending=4_000,
        accounts=[AccountConfig(account_type="taxable", balance=1e6)],
        healthcare=healthcare,
    )
    result = simulate(plan, market, PolicyBundle(), SimulationConfig(n_paths=10))
    return result.terminal_wealth_percentiles["p50"]


class TestHealthcare:
    def test_costs_are_paid_on_top_of_spending(self) -> None:
        healthcare = HealthcareCost(monthly_premium=600, monthly_out_of_pocket=400, growth_rate=0)
        assert _terminal_wealth(healthcare) == pytest.approx(1e6 - 24 * 5_000)

    def test_costs_grow_monthly_at_the_growth_rate(self) -> None:
        healthcare = HealthcareCost(monthly_premium=1_000, growth_rate=0.1)
        growth = 1.1 ** (1 / 12)
        costs = sum(1_000 * growth**t for t in range(24))
        assert _terminal_wealth(healthcare) == pytest.approx(1e6 - 24 * 4_000 - costs)

    def test_costs_start_at_retirement_and_grow_from_today(self) -> None:
        healthcare = HealthcareCost(monthly_premium=1_000, growth_rate=0.1)
        growth = 1.1 ** (1 / 12)
        costs = sum(1_000 * growth**t for t in range(12, 24))
        assert _terminal_wealth(healthcare, retirement_age=61) == pytest.approx(
            1e6 - 12 * 4_000 - costs
        )

    def test_no_healthcare_costs_by_default(self) -> None:
        assert _terminal_wealth(None) == pytest.approx(1e6 - 24 * 4_000)
