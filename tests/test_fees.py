"""Tests for investment fee drag (G2)."""

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate


def _base_plan():
    return PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=70,
        accounts=[AccountConfig(account_type="taxable", balance=1_000_000)],
        monthly_income=0,
        monthly_spending=0,
    )


def _base_market(**overrides):
    defaults = dict(
        asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
        expected_annual_returns=[0.0],
        annual_volatilities=[0.0001],
        correlation_matrix=[[1.0]],
        inflation_mean=0.0,
        inflation_vol=0.0,
    )
    defaults.update(overrides)
    return MarketAssumptions(**defaults)


def test_zero_fees_no_drag():
    """Zero fees should not reduce wealth (baseline)."""
    result = simulate(
        _base_plan(),
        _base_market(expense_ratio=0.0, aum_fee=0.0, advisory_fee=0.0),
        PolicyBundle(spending=PolicyBundle().spending, rebalancing_months=[]),
        SimulationConfig(n_paths=10, seed=1),
    )
    # With near-zero vol and zero returns/fees, wealth ~ initial
    median = result.terminal_wealth_percentiles["p50"]
    assert median > 990_000, f"Expected ~1M, got {median}"


def test_fees_reduce_wealth():
    """Non-zero fees should reduce terminal wealth vs zero fees."""
    plan = _base_plan()
    sim = SimulationConfig(n_paths=100, seed=42)
    pol = PolicyBundle(spending=PolicyBundle().spending, rebalancing_months=[])

    result_no_fee = simulate(plan, _base_market(), pol, sim)
    result_with_fee = simulate(
        plan,
        _base_market(expense_ratio=0.005, aum_fee=0.005, advisory_fee=0.005),
        pol,
        sim,
    )

    median_no_fee = result_no_fee.terminal_wealth_percentiles["p50"]
    median_with_fee = result_with_fee.terminal_wealth_percentiles["p50"]
    assert median_with_fee < median_no_fee, (
        f"Fees should reduce wealth: {median_with_fee} vs {median_no_fee}"
    )


def test_fee_drag_magnitude():
    """1% annual fee over 10 years should reduce wealth by roughly 10%."""
    plan = _base_plan()
    sim = SimulationConfig(n_paths=50, seed=42)
    pol = PolicyBundle(spending=PolicyBundle().spending, rebalancing_months=[])

    result = simulate(
        plan,
        _base_market(expense_ratio=0.01),
        pol,
        sim,
    )
    median = result.terminal_wealth_percentiles["p50"]
    # 1% annual for 10 years: (1-0.01)^10 ≈ 0.904, so ~$904k
    # With near-zero returns, expect ~$900k-$910k
    assert 850_000 < median < 960_000, f"Expected ~904k, got {median}"
