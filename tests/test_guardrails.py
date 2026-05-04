"""Tests for Guyton-Klinger guardrails spending policy."""

from __future__ import annotations

import numpy as np

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    GuardrailsConfig,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
    SpendingPolicyConfig,
)
from monteplan.core.engine import simulate
from monteplan.core.state import SimulationState
from monteplan.policies.spending.guardrails import GuardrailsSpending


def _make_state(wealth: float, n_paths: int = 1) -> SimulationState:
    positions = np.full((n_paths, 1, 1), wealth)
    state = SimulationState(
        positions=positions,
        cumulative_inflation=np.ones(n_paths),
        is_depleted=np.zeros(n_paths, dtype=bool),
        step=0,
        n_paths=n_paths,
        n_accounts=1,
        n_assets=1,
        account_types=["taxable"],
        annual_ordinary_income=np.zeros(n_paths),
        annual_ltcg=np.zeros(n_paths),
        prior_year_traditional_balance=np.zeros(n_paths),
        annual_rmd_satisfied=np.zeros(n_paths),
        current_spending=np.zeros(n_paths),
        initial_portfolio_value=np.full(n_paths, wealth),
    )
    return state


class TestGuardrails:
    def test_initial_spending(self) -> None:
        """First call should initialize spending at initial_withdrawal_rate."""
        config = GuardrailsConfig(initial_withdrawal_rate=0.05)
        policy = GuardrailsSpending(config)
        state = _make_state(1_000_000)
        result = policy.compute(state)
        # 5% of 1M / 12 = 4166.67
        np.testing.assert_allclose(result, [1_000_000 * 0.05 / 12], rtol=0.01)

    def test_capital_preservation_cut(self) -> None:
        """When withdrawal rate is too high, spending should be cut."""
        config = GuardrailsConfig(
            initial_withdrawal_rate=0.05,
            lower_threshold=0.20,
            cut_pct=0.10,
        )
        policy = GuardrailsSpending(config)
        state = _make_state(500_000)  # Wealth dropped
        # Set current spending as if it was 5% of 1M = 4166.67/month
        state.current_spending = np.array([4_166.67])
        # Current rate = 4166.67*12 / 500000 = 10% which > 6% (5%*1.2)
        result = policy.compute(state)
        # Should have been cut by 10%
        np.testing.assert_allclose(result, [4_166.67 * 0.9], rtol=0.01)

    def test_prosperity_raise(self) -> None:
        """When withdrawal rate is too low, spending should increase."""
        config = GuardrailsConfig(
            initial_withdrawal_rate=0.05,
            upper_threshold=0.20,
            raise_pct=0.10,
        )
        policy = GuardrailsSpending(config)
        state = _make_state(2_000_000)
        # Set current spending at 5% of 1M = 4166.67/month
        state.current_spending = np.array([4_166.67])
        # Current rate = 4166.67*12 / 2000000 = 2.5% which < 4% (5%*0.8)
        result = policy.compute(state)
        # Should have been raised by 10%
        np.testing.assert_allclose(result, [4_166.67 * 1.1], rtol=0.01)

    def test_engine_integration(self) -> None:
        """Guardrails policy should run in the full simulation."""
        plan = PlanConfig(
            current_age=60,
            retirement_age=65,
            end_age=95,
            accounts=[AccountConfig(balance=1_000_000)],
            monthly_income=5_000,
            monthly_spending=4_000,
        )
        policies = PolicyBundle(
            spending=SpendingPolicyConfig(policy_type="guardrails"),
        )
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
            expected_annual_returns=[0.07],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
        )
        result = simulate(plan, market, policies, SimulationConfig(n_paths=50, seed=42))
        assert 0.0 <= result.success_probability <= 1.0
