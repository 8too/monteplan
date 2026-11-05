"""Tests for Required Minimum Distributions."""

from __future__ import annotations

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
)
from monteplan.core.engine import simulate
from monteplan.taxes.rmd import RMDCalculator
from monteplan.taxes.us_federal import USFederalTaxModel


def _fixed_market() -> MarketAssumptions:
    """No returns and no inflation, so every path is the same."""
    return MarketAssumptions(
        asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
        expected_annual_returns=[0.0],
        annual_volatilities=[0.0],
        correlation_matrix=[[1.0]],
        inflation_mean=0.0,
        inflation_vol=0.0,
    )


class TestRMDCalculator:
    def test_start_age(self) -> None:
        calc = RMDCalculator()
        assert calc.start_age == 73

    def test_no_rmd_before_73(self) -> None:
        calc = RMDCalculator()
        balance = np.array([1_000_000.0])
        rmd = calc.compute_rmd(72, balance)
        np.testing.assert_allclose(rmd, [0.0])

    def test_rmd_at_73(self) -> None:
        calc = RMDCalculator()
        balance = np.array([1_000_000.0])
        rmd = calc.compute_rmd(73, balance)
        # Divisor at 73 = 26.5
        np.testing.assert_allclose(rmd, [1_000_000.0 / 26.5])

    def test_rmd_at_85(self) -> None:
        calc = RMDCalculator()
        balance = np.array([500_000.0])
        rmd = calc.compute_rmd(85, balance)
        # Divisor at 85 = 16.0
        np.testing.assert_allclose(rmd, [500_000.0 / 16.0])

    def test_divisor_lookup(self) -> None:
        calc = RMDCalculator()
        assert calc.divisor(73) == 26.5
        assert calc.divisor(80) == 20.2
        assert calc.divisor(90) == 12.2

    def test_vectorized(self) -> None:
        calc = RMDCalculator()
        balance = np.array([1_000_000.0, 500_000.0, 200_000.0])
        rmd = calc.compute_rmd(75, balance)
        # Divisor at 75 = 24.6
        expected = balance / 24.6
        np.testing.assert_allclose(rmd, expected)


class TestRMDEngineIntegration:
    def test_rmd_forces_withdrawal(self) -> None:
        """RMDs should force withdrawals from traditional accounts after age 73."""
        # Person with large traditional balance, minimal spending
        plan = PlanConfig(
            current_age=70,
            retirement_age=71,
            end_age=80,
            accounts=[
                AccountConfig(account_type="traditional", balance=2_000_000),
            ],
            monthly_income=0,
            monthly_spending=1_000,  # Very low spending
        )
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
            expected_annual_returns=[0.0],
            annual_volatilities=[0.001],
            correlation_matrix=[[1.0]],
            inflation_mean=0.0,
            inflation_vol=0.0001,
        )
        sim = SimulationConfig(n_paths=10, seed=42, store_paths=True)
        result = simulate(plan, market, PolicyBundle(tax_model="us_federal"), sim)

        # With $2M and minimal spending, RMDs should force significant withdrawals
        # after age 73 (step = (73-70)*12 = 36)
        assert result.all_paths is not None
        # Wealth should decrease due to RMDs even though spending is minimal
        # Check that terminal wealth is lower than what it would be with just $1k/mo spending
        # $1k/mo for 10 years = $120k total spending, so without RMDs ~$1.88M
        # With RMDs + taxes, should be noticeably lower
        median_terminal = float(np.median(result.all_paths[:, -1]))
        assert median_terminal < 1_900_000  # RMDs + taxes took a chunk


class TestRMDDistribution:
    """One year at 75 with no spending: the whole RMD is distributed and reinvested."""

    RMD = 1_000_000 / 24.6  # Uniform Lifetime Table divisor at 75

    @staticmethod
    def _terminal_wealth(tax_model: str, accounts: list[AccountConfig]) -> float:
        plan = PlanConfig(
            current_age=75, retirement_age=75, end_age=76, monthly_spending=0, accounts=accounts
        )
        market = _fixed_market()
        if len(accounts) > 1:
            market = market.model_copy(
                update={"asset_allocations": market.asset_allocations * len(accounts)}
            )
        result = simulate(
            plan,
            market,
            PolicyBundle(tax_model=tax_model),  # type: ignore[arg-type]
            SimulationConfig(n_paths=4),
        )
        return result.terminal_wealth_percentiles["p50"]

    def test_reinvested_net_of_flat_tax(self) -> None:
        """Also checks that the first year's RMD uses the starting balance."""
        wealth = self._terminal_wealth(
            "flat", [AccountConfig(account_type="traditional", balance=1e6)]
        )
        assert wealth == pytest.approx(1e6 - 0.22 * self.RMD)

    def test_reinvested_and_taxed_at_year_end_under_us_federal(self) -> None:
        wealth = self._terminal_wealth(
            "us_federal", [AccountConfig(account_type="traditional", balance=1e6)]
        )
        tax = USFederalTaxModel().compute_annual_tax(self.RMD, 0, "single")
        assert wealth == pytest.approx(1e6 - tax)

    def test_reinvested_in_existing_taxable_account(self) -> None:
        wealth = self._terminal_wealth(
            "flat",
            [
                AccountConfig(account_type="traditional", balance=1e6),
                AccountConfig(account_type="taxable", balance=1e5),
            ],
        )
        assert wealth == pytest.approx(1.1e6 - 0.22 * self.RMD)

    def test_rmds_recur_every_year_under_flat_tax(self) -> None:
        plan = PlanConfig(
            current_age=75,
            retirement_age=75,
            end_age=78,
            monthly_spending=0,
            accounts=[AccountConfig(account_type="traditional", balance=1e6)],
        )
        result = simulate(
            plan,
            _fixed_market(),
            PolicyBundle(tax_model="flat"),
            SimulationConfig(n_paths=4, store_paths=True),
        )
        assert result.all_paths is not None
        year_end_wealth = result.all_paths[0, [0, 12, 24, 36]]
        # Each year loses the flat tax on that year's RMD.
        assert np.all(np.diff(year_end_wealth) < -1_000)
