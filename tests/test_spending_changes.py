"""Tests for step changes to retirement spending by age (e.g. after a spouse dies)."""

import numpy as np
import pytest

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
    SpendingChange,
    SpendingPolicyConfig,
)
from monteplan.core.engine import simulate


def _zero_vol_market():
    return MarketAssumptions(
        asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
        expected_annual_returns=[0.0],
        annual_volatilities=[0.0001],
        correlation_matrix=[[1.0]],
        inflation_mean=0.0,
        inflation_vol=0.0,
    )


def _run(spending_changes, policy_type="constant_real"):
    plan = PlanConfig(
        current_age=64,
        retirement_age=65,
        end_age=85,
        accounts=[AccountConfig(account_type="taxable", balance=1_000_000)],
        monthly_spending=4000,
        spending_changes=spending_changes,
    )
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type=policy_type), rebalancing_months=[]
    )
    return simulate(plan, _zero_vol_market(), pol, SimulationConfig(n_paths=50, seed=7))


def _monthly_spending_at(result, age):
    step = int(round((age - 64) * 12))
    return float(result.spending_time_series["p50"][step])


def test_step_lowers_spending_from_its_age_only():
    base = _run([])
    cut = _run([SpendingChange(age=75, multiplier=0.75)])
    assert _monthly_spending_at(cut, 70) == pytest.approx(_monthly_spending_at(base, 70))
    assert _monthly_spending_at(cut, 76) == pytest.approx(
        0.75 * _monthly_spending_at(base, 76), rel=1e-6
    )
    assert cut.terminal_wealth_percentiles["p50"] > base.terminal_wealth_percentiles["p50"]


def test_multiplier_of_one_changes_nothing():
    base = _run([])
    same = _run([SpendingChange(age=75, multiplier=1.0)])
    assert same.terminal_wealth_percentiles == base.terminal_wealth_percentiles
    np.testing.assert_allclose(same.spending_time_series["p50"], base.spending_time_series["p50"])


def test_steps_compound():
    two = _run([SpendingChange(age=70, multiplier=0.9), SpendingChange(age=75, multiplier=0.9)])
    base = _run([])
    assert _monthly_spending_at(two, 76) == pytest.approx(
        0.81 * _monthly_spending_at(base, 76), rel=1e-6
    )


def test_step_applies_to_other_spending_policies():
    base = _run([], "percent_of_portfolio")
    cut = _run([SpendingChange(age=75, multiplier=0.5)], "percent_of_portfolio")
    assert _monthly_spending_at(cut, 70) == pytest.approx(_monthly_spending_at(base, 70))
    assert _monthly_spending_at(cut, 76) < _monthly_spending_at(base, 76)
