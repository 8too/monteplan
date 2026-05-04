"""Tests for Variable Percentage Withdrawal spending policy."""

from __future__ import annotations

import numpy as np

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
    SpendingPolicyConfig,
    VPWConfig,
)
from monteplan.core.engine import simulate
from monteplan.core.state import SimulationState
from monteplan.policies.spending.vpw import VPWSpending


def _make_state(wealth: float, step: int = 0) -> SimulationState:
    positions = np.array([[[wealth]]])
    return SimulationState(
        positions=positions,
        cumulative_inflation=np.ones(1),
        is_depleted=np.zeros(1, dtype=bool),
        step=step,
        n_paths=1,
        n_accounts=1,
        n_assets=1,
        account_types=["taxable"],
        annual_ordinary_income=np.zeros(1),
        annual_ltcg=np.zeros(1),
        prior_year_traditional_balance=np.zeros(1),
        annual_rmd_satisfied=np.zeros(1),
        current_spending=np.zeros(1),
        initial_portfolio_value=np.array([wealth]),
    )


class TestVPW:
    def test_early_retirement_low_rate(self) -> None:
        """Early in retirement with many years left, rate should be low."""
        config = VPWConfig(min_rate=0.03, max_rate=0.15)
        policy = VPWSpending(config, end_age=95, current_age=65)
        state = _make_state(1_000_000, step=0)  # age 65, 30 years left
        result = policy.compute(state)
        # Rate = 1/30 = 0.0333, monthly = 0.0333/12 * 1M = 2778
        np.testing.assert_allclose(result, [1_000_000 * (1.0 / 30.0) / 12], rtol=0.01)

    def test_late_retirement_high_rate(self) -> None:
        """Late in retirement, rate increases (but bounded by max_rate)."""
        config = VPWConfig(min_rate=0.03, max_rate=0.15)
        policy = VPWSpending(config, end_age=95, current_age=65)
        # step = 25 * 12 = 300 → age 90, 5 years left
        state = _make_state(500_000, step=300)
        result = policy.compute(state)
        # Rate = 1/5 = 0.20, bounded to 0.15
        np.testing.assert_allclose(result, [500_000 * 0.15 / 12], rtol=0.01)

    def test_min_rate_bound(self) -> None:
        """Rate should never go below min_rate."""
        config = VPWConfig(min_rate=0.04, max_rate=0.15)
        policy = VPWSpending(config, end_age=95, current_age=30)
        state = _make_state(1_000_000, step=0)  # age 30, 65 years left
        result = policy.compute(state)
        # Rate = 1/65 = 0.0154, bounded to 0.04
        np.testing.assert_allclose(result, [1_000_000 * 0.04 / 12], rtol=0.01)

    def test_zero_wealth(self) -> None:
        """Should return zero when wealth is zero."""
        config = VPWConfig()
        policy = VPWSpending(config, end_age=95, current_age=65)
        state = _make_state(0.0)
        result = policy.compute(state)
        np.testing.assert_allclose(result, [0.0])

    def test_engine_integration(self) -> None:
        """VPW policy should run in the full simulation."""
        plan = PlanConfig(
            current_age=60,
            retirement_age=65,
            end_age=95,
            accounts=[AccountConfig(balance=1_000_000)],
            monthly_income=5_000,
            monthly_spending=4_000,
        )
        policies = PolicyBundle(
            spending=SpendingPolicyConfig(policy_type="vpw"),
        )
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
            expected_annual_returns=[0.07],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
        )
        result = simulate(plan, market, policies, SimulationConfig(n_paths=50, seed=42))
        # VPW never depletes (it scales with portfolio)
        assert result.success_probability > 0.5
