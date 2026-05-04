"""Tests for threshold-based rebalancing (G3)."""

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
)
from monteplan.core.engine import simulate
from monteplan.core.state import SimulationState
from monteplan.policies.rebalancing import rebalance_if_drifted


def _make_state(n_paths, positions_array, account_types=None):
    """Create a SimulationState from a positions array."""
    if account_types is None:
        account_types = ["taxable"]
    n_accounts = positions_array.shape[1]
    n_assets = positions_array.shape[2]
    state = SimulationState(
        positions=positions_array.copy(),
        cumulative_inflation=np.ones(n_paths),
        is_depleted=np.zeros(n_paths, dtype=bool),
        step=0,
        n_paths=n_paths,
        n_accounts=n_accounts,
        n_assets=n_assets,
        account_types=account_types,
        annual_ordinary_income=np.zeros(n_paths),
        annual_ltcg=np.zeros(n_paths),
        prior_year_traditional_balance=np.zeros(n_paths),
        annual_rmd_satisfied=np.zeros(n_paths),
        current_spending=np.zeros(n_paths),
        initial_portfolio_value=np.zeros(n_paths),
    )
    return state


def test_no_rebalance_within_threshold():
    """Positions within threshold should not be touched."""
    # 1 path, 1 account, 2 assets: 68% / 32% with target 70/30
    positions = np.array([[[68_000.0, 32_000.0]]])
    state = _make_state(1, positions)
    target = np.array([[0.7, 0.3]])

    original = state.positions.copy()
    rebalance_if_drifted(state, target, threshold=0.05)

    np.testing.assert_array_equal(state.positions, original)


def test_rebalance_when_drifted():
    """Positions beyond threshold should be rebalanced."""
    # 1 path, 1 account, 2 assets: 80% / 20% with target 70/30 → drift 10%
    positions = np.array([[[80_000.0, 20_000.0]]])
    state = _make_state(1, positions)
    target = np.array([[0.7, 0.3]])

    rebalance_if_drifted(state, target, threshold=0.05)

    # Total = 100k, should be 70k/30k
    np.testing.assert_allclose(state.positions[0, 0], [70_000, 30_000], atol=1)


def test_mixed_paths_selective_rebalance():
    """Only paths that drifted should be rebalanced."""
    # Path 0: 68/32 (within 5%) → no rebalance
    # Path 1: 82/18 (beyond 5%) → rebalance
    positions = np.array(
        [
            [[68_000.0, 32_000.0]],
            [[82_000.0, 18_000.0]],
        ]
    )
    state = _make_state(2, positions)
    target = np.array([[0.7, 0.3]])

    rebalance_if_drifted(state, target, threshold=0.05)

    # Path 0 unchanged
    np.testing.assert_allclose(state.positions[0, 0], [68_000, 32_000], atol=1)
    # Path 1 rebalanced
    np.testing.assert_allclose(state.positions[1, 0], [70_000, 30_000], atol=1)


def test_threshold_rebalancing_integration():
    """End-to-end simulation with threshold rebalancing should run."""
    plan = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=70,
        accounts=[AccountConfig(account_type="taxable", balance=500_000)],
        monthly_income=0,
        monthly_spending=3000,
    )
    market = MarketAssumptions(
        asset_allocations=[
            AssetAllocation(
                assets=[
                    AssetClass(name="Stocks", weight=0.7),
                    AssetClass(name="Bonds", weight=0.3),
                ]
            )
        ],
        expected_annual_returns=[0.07, 0.03],
        annual_volatilities=[0.16, 0.06],
        correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        inflation_mean=0.03,
        inflation_vol=0.01,
    )
    policies = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_strategy="threshold",
        rebalancing_threshold=0.05,
    )
    sim = SimulationConfig(n_paths=100, seed=42)

    result = simulate(plan, market, policies, sim)
    assert 0.0 <= result.success_probability <= 1.0
