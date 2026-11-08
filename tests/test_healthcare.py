"""Tests for healthcare costs in retirement."""

from __future__ import annotations

import pytest

from monteplan.config.defaults import default_market
from monteplan.config.schema import (
    AccountConfig,
    HealthcareCost,
    LotConfig,
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


def _zero_basis_wealth(
    healthcare: HealthcareCost,
    *,
    current_age: int = 60,
    end_age: int = 61,
    monthly_spending: float = 4_000,
    filing_status: str = "single",
) -> float:
    """Terminal wealth of a zero-basis taxable account without returns, inflation or tax.

    Every dollar withdrawn is a realized gain, so a year's income is what it withdrew.
    """
    market = default_market().model_copy(
        update={
            "expected_annual_returns": [0.0] * 6,
            "annual_volatilities": [0.0] * 6,
            "inflation_mean": 0.0,
            "inflation_vol": 0.0,
        }
    )
    plan = PlanConfig(
        current_age=current_age,
        retirement_age=current_age,
        end_age=end_age,
        monthly_spending=monthly_spending,
        accounts=[
            AccountConfig(
                account_type="taxable",
                balance=1e6,
                lots=[LotConfig(amount=1e6, cost_basis=0.0, months_held=24)],
            )
        ],
        healthcare=healthcare,
    )
    # The flat tax model doesn't tax gains, so only the premium tax credit is settled.
    policies = PolicyBundle(filing_status=filing_status)  # type: ignore[arg-type]
    result = simulate(plan, market, policies, SimulationConfig(n_paths=10))
    return result.terminal_wealth_percentiles["p50"]


class TestPremiumTaxCredit:
    def test_credit_paid_at_year_end(self) -> None:
        healthcare = HealthcareCost(
            monthly_premium=1_000, monthly_benchmark_premium=900, growth_rate=0
        )
        # Income = 12 * (4,000 + 1,000) = 60,000, 383% FPL: contribution 9.96% = 5,976;
        # credit 12 * 900 - 5,976 = 4,824
        assert _zero_basis_wealth(healthcare) == pytest.approx(1e6 - 60_000 + 4_824)

    def test_no_credit_above_400_percent(self) -> None:
        healthcare = HealthcareCost(
            monthly_premium=1_000, monthly_benchmark_premium=900, growth_rate=0
        )
        # Income = 12 * 6,000 = 72,000, above 400% FPL (62,600)
        wealth = _zero_basis_wealth(healthcare, monthly_spending=5_000)
        assert wealth == pytest.approx(1e6 - 72_000)

    def test_enhanced_credit_above_400_percent(self) -> None:
        healthcare = HealthcareCost(
            monthly_premium=1_000,
            monthly_benchmark_premium=900,
            aca_subsidy="enhanced",
            growth_rate=0,
        )
        # 8.5% of 72,000 = 6,120; credit 10,800 - 6,120 = 4,680
        wealth = _zero_basis_wealth(healthcare, monthly_spending=5_000)
        assert wealth == pytest.approx(1e6 - 72_000 + 4_680)

    def test_no_credit_without_subsidy(self) -> None:
        healthcare = HealthcareCost(
            monthly_premium=1_000, monthly_benchmark_premium=900, aca_subsidy="none", growth_rate=0
        )
        assert _zero_basis_wealth(healthcare) == pytest.approx(1e6 - 60_000)

    def test_credit_follows_the_income(self) -> None:
        """Without gains the income is zero and the credit the full benchmark premium."""
        healthcare = HealthcareCost(
            monthly_premium=1_000, monthly_benchmark_premium=900, growth_rate=0
        )
        assert _terminal_wealth(healthcare) == pytest.approx(1e6 - 24 * 5_000 + 2 * 10_800)


class TestMedicare:
    def test_switches_from_aca_to_medicare(self) -> None:
        healthcare = HealthcareCost(
            monthly_premium=1_000, monthly_medicare_supplement=300, growth_rate=0
        )
        wealth = _zero_basis_wealth(healthcare, current_age=64, end_age=66)
        # Age 64: the ACA plan; from 65: Part B plus the supplement
        assert wealth == pytest.approx(1e6 - 24 * 4_000 - 12 * 1_000 - 12 * (202.90 + 300))

    def test_part_b_for_each_enrollee(self) -> None:
        healthcare = HealthcareCost(
            household_size=2, monthly_medicare_supplement=500, growth_rate=0
        )
        wealth = _zero_basis_wealth(
            healthcare,
            current_age=65,
            end_age=66,
            monthly_spending=1_000,
            filing_status="married_jointly",
        )
        assert wealth == pytest.approx(1e6 - 12 * (1_000 + 2 * 202.90 + 500))

    def test_costs_grow_from_today(self) -> None:
        healthcare = HealthcareCost(monthly_medicare_supplement=300, growth_rate=0.1)
        growth = 1.1 ** (1 / 12)
        costs = sum((202.90 + 300) * growth**t for t in range(12))
        wealth = _zero_basis_wealth(
            healthcare, current_age=65, end_age=66, monthly_spending=1_000
        )
        assert wealth == pytest.approx(1e6 - 12 * 1_000 - costs)

    def test_irmaa_from_income_two_years_earlier(self) -> None:
        healthcare = HealthcareCost(growth_rate=0)
        # Income is 12 * 12,000 = 144,000 every year: the 137,000-171,000 tier. Medicare
        # starts in the third year, whose surcharges come from the first year's income.
        wealth = _zero_basis_wealth(
            healthcare, current_age=63, end_age=66, monthly_spending=12_000
        )
        assert wealth == pytest.approx(1e6 - 36 * 12_000 - 12 * (202.90 + 202.90 + 37.50))

    def test_no_irmaa_for_income_before_the_start(self) -> None:
        healthcare = HealthcareCost(growth_rate=0)
        # The first two years' surcharges would depend on income before the simulation
        wealth = _zero_basis_wealth(
            healthcare, current_age=65, end_age=67, monthly_spending=12_000
        )
        assert wealth == pytest.approx(1e6 - 24 * 12_000 - 24 * 202.90)

    def test_married_irmaa_thresholds(self) -> None:
        healthcare = HealthcareCost(household_size=2, growth_rate=0)
        # 144,000 is below the married filing jointly threshold of 218,000
        wealth = _zero_basis_wealth(
            healthcare,
            current_age=63,
            end_age=66,
            monthly_spending=12_000,
            filing_status="married_jointly",
        )
        assert wealth == pytest.approx(1e6 - 36 * 12_000 - 12 * 2 * 202.90)
