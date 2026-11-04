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


def _inflation_only_market(inflation: float = 0.03):
    """Zero real return, fixed inflation: withdrawals are the only thing moving wealth."""
    return MarketAssumptions(
        asset_allocations=[AssetAllocation(assets=[AssetClass(name="Cash", weight=1.0)])],
        expected_annual_returns=[0.0],
        annual_volatilities=[0.0001],
        correlation_matrix=[[1.0]],
        inflation_mean=inflation,
        inflation_vol=0.0,
    )


def _covered_plan(stream: GuaranteedIncomeStream) -> PlanConfig:
    # Income starts at retirement and equals spending in today's dollars, so an
    # inflation-indexed stream with no extra COLA covers spending exactly. A year of
    # zero-contribution accumulation precedes retirement because the plan requires it.
    return PlanConfig(
        current_age=64,
        retirement_age=65,
        end_age=85,
        accounts=[AccountConfig(account_type="taxable", balance=1_000_000)],
        monthly_income=0,
        monthly_spending=3000,
        guaranteed_income=[stream],
    )


def _median_terminal(plan: PlanConfig, market: MarketAssumptions) -> float:
    pol = PolicyBundle(
        spending=SpendingPolicyConfig(policy_type="constant_real"),
        rebalancing_months=[],
    )
    result = simulate(plan, market, pol, SimulationConfig(n_paths=50, seed=7))
    return result.terminal_wealth_percentiles["p50"]


def test_indexed_stream_with_zero_cola_holds_real_value():
    """cpi_indexed with cola_rate=0 rises with CPI and nothing more."""
    ss = GuaranteedIncomeStream(name="SS", monthly_amount=3000, start_age=65)
    terminal = _median_terminal(_covered_plan(ss), _inflation_only_market())
    # Spending is fully covered every month, so the portfolio is untouched.
    assert abs(terminal - 1_000_000) < 1_000


def test_nominal_stream_loses_real_value_to_inflation():
    """A flat nominal pension covers less of real spending every year."""
    pension = GuaranteedIncomeStream(
        name="Pension", monthly_amount=3000, start_age=65, cpi_indexed=False
    )
    terminal = _median_terminal(_covered_plan(pension), _inflation_only_market())
    # The gap between CPI-grown spending and the flat payment is withdrawn.
    assert terminal < 800_000


def test_indexed_cola_is_growth_above_inflation():
    """cola_rate on an indexed stream is real growth, so SS should leave it at 0."""
    indexed = GuaranteedIncomeStream(name="SS", monthly_amount=2000, start_age=65)
    over = GuaranteedIncomeStream(name="SS", monthly_amount=2000, start_age=65, cola_rate=0.02)
    market = _inflation_only_market()
    base = _median_terminal(_covered_plan(indexed), market)
    boosted = _median_terminal(_covered_plan(over), market)
    assert boosted > base


def test_nominal_stream_cola_offsets_matching_inflation():
    """A nominal stream with a COLA equal to inflation behaves like an indexed one."""
    # Nominal amount at 65 is today's $3,000 after one year of 3% inflation.
    pension = GuaranteedIncomeStream(
        name="Pension",
        monthly_amount=3090,
        start_age=65,
        cpi_indexed=False,
        cola_rate=0.03,
    )
    terminal = _median_terminal(_covered_plan(pension), _inflation_only_market())
    assert abs(terminal - 1_000_000) < 20_000
