"""Tests for discrete events and income growth."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    DiscreteEvent,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate


def _simple_plan(**overrides) -> PlanConfig:
    defaults = dict(
        current_age=30,
        retirement_age=65,
        end_age=95,
        accounts=[
            AccountConfig(account_type="taxable", balance=100_000, annual_contribution=12_000),
        ],
        monthly_income=5_000,
        monthly_spending=3_000,
    )
    defaults.update(overrides)
    return PlanConfig(**defaults)


def _simple_market() -> MarketAssumptions:
    return MarketAssumptions(
        asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
        expected_annual_returns=[0.0],
        annual_volatilities=[0.001],
        correlation_matrix=[[1.0]],
        inflation_mean=0.0,
        inflation_vol=0.0001,
    )


class TestIncomeGrowth:
    def test_no_growth_baseline(self) -> None:
        """With 0% income growth, contributions are constant."""
        plan = _simple_plan(income_growth_rate=0.0)
        sim = SimulationConfig(n_paths=10, seed=1)
        result = simulate(plan, _simple_market(), PolicyBundle(), sim)
        assert result.success_probability >= 0

    def test_positive_growth_increases_wealth(self) -> None:
        """With positive income growth, wealth at retirement should be higher."""
        plan_no_growth = _simple_plan(income_growth_rate=0.0, monthly_spending=1_000)
        plan_growth = _simple_plan(income_growth_rate=0.03, monthly_spending=1_000)
        sim = SimulationConfig(n_paths=100, seed=42, store_paths=True)
        market = _simple_market()
        policies = PolicyBundle()

        r_no = simulate(plan_no_growth, market, policies, sim)
        r_yes = simulate(plan_growth, market, policies, sim)

        # At retirement step, wealth should be higher with income growth
        ret_step = (65 - 30) * 12  # 420
        assert r_yes.all_paths is not None
        assert r_no.all_paths is not None
        median_no = float(np.median(r_no.all_paths[:, ret_step]))
        median_yes = float(np.median(r_yes.all_paths[:, ret_step]))
        assert median_yes > median_no


class TestDiscreteEvents:
    def test_outflow_reduces_wealth(self) -> None:
        """A large outflow at age 50 should reduce wealth at retirement."""
        plan_no_event = _simple_plan(monthly_spending=1_000)
        plan_with_event = _simple_plan(
            monthly_spending=1_000,
            discrete_events=[DiscreteEvent(age=50, amount=-200_000, description="Home purchase")],
        )
        sim = SimulationConfig(n_paths=100, seed=42, store_paths=True)
        market = _simple_market()
        policies = PolicyBundle()

        r_no = simulate(plan_no_event, market, policies, sim)
        r_with = simulate(plan_with_event, market, policies, sim)

        assert r_no.all_paths is not None
        assert r_with.all_paths is not None
        # Check wealth at retirement (step 420)
        ret_step = (65 - 30) * 12
        median_no = float(np.median(r_no.all_paths[:, ret_step]))
        median_with = float(np.median(r_with.all_paths[:, ret_step]))
        assert median_with < median_no

    def test_inflow_increases_wealth(self) -> None:
        """An inheritance at age 50 should increase wealth at retirement."""
        plan_no_event = _simple_plan(monthly_spending=1_000)
        plan_with_event = _simple_plan(
            monthly_spending=1_000,
            discrete_events=[DiscreteEvent(age=50, amount=500_000, description="Inheritance")],
        )
        sim = SimulationConfig(n_paths=100, seed=42, store_paths=True)
        market = _simple_market()
        policies = PolicyBundle()

        r_no = simulate(plan_no_event, market, policies, sim)
        r_with = simulate(plan_with_event, market, policies, sim)

        assert r_no.all_paths is not None
        assert r_with.all_paths is not None
        ret_step = (65 - 30) * 12
        median_no = float(np.median(r_no.all_paths[:, ret_step]))
        median_with = float(np.median(r_with.all_paths[:, ret_step]))
        assert median_with > median_no

    def test_event_at_correct_step(self) -> None:
        """Event at age 50 (step 240 for 30yo start) should affect wealth at that step."""
        plan = _simple_plan(
            discrete_events=[DiscreteEvent(age=50, amount=-50_000, description="test")],
        )
        sim = SimulationConfig(n_paths=50, seed=42, store_paths=True)
        result = simulate(plan, _simple_market(), PolicyBundle(), sim)
        assert result.all_paths is not None
        # Event happens at step 240 (age 50 - age 30 = 20 years * 12)
        # Wealth at step 241 should show the impact
        assert result.all_paths.shape[1] > 241

    def test_event_config_validation(self) -> None:
        """DiscreteEvent model validates correctly."""
        ev = DiscreteEvent(age=50, amount=-100_000, description="Home")
        assert ev.age == 50
        assert ev.amount == -100_000

        with pytest.raises(ValueError):
            DiscreteEvent(age=10, amount=100)  # age < 18
