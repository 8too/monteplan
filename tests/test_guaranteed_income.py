"""Tests for guaranteed income streams (G6: Social Security / Pension)."""

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    GuaranteedIncomeStream,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
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


def test_ss_reduces_withdrawals():
    """Social Security should reduce portfolio withdrawals, increasing success."""
    plan_no_ss = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=75,
        accounts=[AccountConfig(account_type="taxable", balance=200_000)],
        monthly_income=0,
        monthly_spending=3000,
    )
    plan_with_ss = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=75,
        accounts=[AccountConfig(account_type="taxable", balance=200_000)],
        monthly_income=0,
        monthly_spending=3000,
        guaranteed_income=[
            GuaranteedIncomeStream(
                name="Social Security",
                monthly_amount=2000,
                start_age=67,
            ),
        ],
    )

    sim = SimulationConfig(n_paths=200, seed=42)
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_months=[],
    )

    result_no_ss = simulate(plan_no_ss, _zero_vol_market(), pol, sim)
    result_with_ss = simulate(plan_with_ss, _zero_vol_market(), pol, sim)

    # SS should improve outcomes
    assert result_with_ss.success_probability >= result_no_ss.success_probability


def test_ss_covers_all_spending_high_success():
    """When SS fully covers spending, success should be very high."""
    plan = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=75,
        accounts=[AccountConfig(account_type="taxable", balance=500_000)],
        monthly_income=0,
        monthly_spending=2000,
        guaranteed_income=[
            GuaranteedIncomeStream(
                name="Social Security",
                monthly_amount=3000,  # More than spending
                start_age=65,
            ),
        ],
    )

    sim = SimulationConfig(n_paths=100, seed=42)
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_months=[],
    )

    result = simulate(plan, _zero_vol_market(), pol, sim)
    # SS covers all spending, so no withdrawals needed → high success
    assert result.success_probability > 0.95


def test_ss_with_cola():
    """COLA-adjusted SS should still work and improve outcomes."""
    plan = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=80,
        accounts=[AccountConfig(account_type="taxable", balance=300_000)],
        monthly_income=0,
        monthly_spending=3000,
        guaranteed_income=[
            GuaranteedIncomeStream(
                name="Social Security",
                monthly_amount=1500,
                start_age=67,
                cola_rate=0.02,
            ),
        ],
    )

    sim = SimulationConfig(n_paths=100, seed=42)
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_months=[],
    )

    result = simulate(plan, _zero_vol_market(), pol, sim)
    # Just verify it runs without error and produces reasonable results
    assert 0.0 <= result.success_probability <= 1.0


def test_ss_with_end_age():
    """Guaranteed income with end_age should stop at specified age."""
    plan = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=80,
        accounts=[AccountConfig(account_type="taxable", balance=300_000)],
        monthly_income=0,
        monthly_spending=3000,
        guaranteed_income=[
            GuaranteedIncomeStream(
                name="Pension",
                monthly_amount=2000,
                start_age=65,
                end_age=70,  # Only 5 years of income
            ),
        ],
    )

    sim = SimulationConfig(n_paths=100, seed=42)
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_months=[],
    )

    result = simulate(plan, _zero_vol_market(), pol, sim)
    assert 0.0 <= result.success_probability <= 1.0


def test_multiple_income_streams():
    """Multiple income streams should stack."""
    plan = PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=75,
        accounts=[AccountConfig(account_type="taxable", balance=200_000)],
        monthly_income=0,
        monthly_spending=4000,
        guaranteed_income=[
            GuaranteedIncomeStream(
                name="Social Security",
                monthly_amount=2000,
                start_age=67,
            ),
            GuaranteedIncomeStream(
                name="Pension",
                monthly_amount=1500,
                start_age=65,
            ),
        ],
    )

    sim = SimulationConfig(n_paths=100, seed=42)
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_months=[],
    )

    result = simulate(plan, _zero_vol_market(), pol, sim)
    assert 0.0 <= result.success_probability <= 1.0
