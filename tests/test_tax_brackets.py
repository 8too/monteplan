"""Tests for US federal bracket-based tax model."""

from __future__ import annotations

import pytest

from monteplan.taxes.us_federal import USFederalTaxModel


@pytest.fixture
def tax_model() -> USFederalTaxModel:
    return USFederalTaxModel(tax_year=2024)


class TestUSFederalTax:
    def test_standard_deduction_single(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.standard_deduction("single") == 14600

    def test_standard_deduction_married(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.standard_deduction("married_jointly") == 29200

    def test_zero_income(self, tax_model: USFederalTaxModel) -> None:
        """No income should produce zero tax."""
        assert tax_model.compute_annual_tax(0, 0, "single") == 0.0

    def test_below_standard_deduction(self, tax_model: USFederalTaxModel) -> None:
        """Income below standard deduction should produce zero ordinary tax."""
        tax = tax_model.compute_annual_tax(10_000, 0, "single")
        assert tax == 0.0

    def test_single_50k_income(self, tax_model: USFederalTaxModel) -> None:
        """Known bracket calculation for $50k single filer."""
        # Taxable = 50000 - 14600 = 35400
        # 10% on first 11600 = 1160
        # 12% on next 23800 (35400 - 11600) = 2856
        # Total = 4016
        tax = tax_model.compute_annual_tax(50_000, 0, "single")
        assert tax == pytest.approx(4016.0, abs=1.0)

    def test_single_100k_income(self, tax_model: USFederalTaxModel) -> None:
        """Known bracket calculation for $100k single filer."""
        # Taxable = 100000 - 14600 = 85400
        # 10% on 11600 = 1160
        # 12% on (47150-11600) = 35550 * 0.12 = 4266
        # 22% on (85400-47150) = 38250 * 0.22 = 8415
        # Total = 13841
        tax = tax_model.compute_annual_tax(100_000, 0, "single")
        assert tax == pytest.approx(13841.0, abs=1.0)

    def test_married_100k_income(self, tax_model: USFederalTaxModel) -> None:
        """Married filing jointly has wider brackets."""
        # Taxable = 100000 - 29200 = 70800
        # 10% on 23200 = 2320
        # 12% on (70800-23200) = 47600 * 0.12 = 5712
        # Total = 8032
        tax = tax_model.compute_annual_tax(100_000, 0, "married_jointly")
        assert tax == pytest.approx(8032.0, abs=1.0)

    def test_ltcg_only(self, tax_model: USFederalTaxModel) -> None:
        """LTCG within 0% bracket should produce zero tax."""
        tax = tax_model.compute_annual_tax(0, 40_000, "single")
        assert tax == 0.0

    def test_ltcg_in_15_percent_bracket(self, tax_model: USFederalTaxModel) -> None:
        """LTCG above 0% threshold should be taxed at 15%."""
        # Single: 0% up to 47025, then 15%
        # 100k LTCG: 0% on 47025 + 15% on 52975 = 7946.25
        tax = tax_model.compute_annual_tax(0, 100_000, "single")
        assert tax == pytest.approx(7946.25, abs=1.0)

    def test_combined_income_and_ltcg(self, tax_model: USFederalTaxModel) -> None:
        """Both ordinary income and LTCG should be taxed."""
        ord_tax = tax_model.compute_annual_tax(50_000, 0, "single")
        ltcg_tax = tax_model.compute_annual_tax(0, 50_000, "single")
        combined_tax = tax_model.compute_annual_tax(50_000, 50_000, "single")
        # Combined should equal sum (in simplified model)
        assert combined_tax == pytest.approx(ord_tax + ltcg_tax, abs=1.0)


class TestMarginalRate:
    def test_lowest_bracket(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.marginal_rate(5_000, "single") == 0.10

    def test_second_bracket(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.marginal_rate(20_000, "single") == 0.12

    def test_22_percent_bracket(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.marginal_rate(80_000, "single") == 0.22

    def test_top_bracket(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.marginal_rate(1_000_000, "single") == 0.37


class TestTaxOnIncome:
    def test_simple_income(self, tax_model: USFederalTaxModel) -> None:
        """tax_on_income uses single filing status."""
        tax = tax_model.tax_on_income(50_000)
        assert tax > 0
        assert tax == pytest.approx(4016.0, abs=1.0)


class TestEngineIntegration:
    def test_us_federal_tax_model_runs(self) -> None:
        """Engine should run with us_federal tax model."""
        from monteplan.config.defaults import default_market, default_plan
        from monteplan.config.schema import PolicyBundle, SimulationConfig
        from monteplan.core.engine import simulate

        plan = default_plan()
        policies = PolicyBundle(tax_model="us_federal", filing_status="single")
        result = simulate(
            plan,
            default_market(),
            policies,
            SimulationConfig(n_paths=50, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0

    def test_flat_vs_federal_differ(self) -> None:
        """Flat and US federal tax models should produce different results."""
        from monteplan.config.defaults import default_market, default_plan
        from monteplan.config.schema import PolicyBundle, SimulationConfig
        from monteplan.core.engine import simulate

        plan = default_plan()
        sim = SimulationConfig(n_paths=200, seed=42)
        market = default_market()

        r_flat = simulate(plan, market, PolicyBundle(tax_model="flat"), sim)
        r_fed = simulate(plan, market, PolicyBundle(tax_model="us_federal"), sim)

        # They should differ (federal has standard deduction + progressive rates). The
        # success probabilities of only 200 paths can coincide, so compare terminal wealth.
        assert (
            r_flat.terminal_wealth_percentiles["p50"] != r_fed.terminal_wealth_percentiles["p50"]
        )

    def test_us_federal_taxes_traditional_withdrawals_once(self) -> None:
        """Withdrawals aren't grossed up for tax that is also charged at year end."""
        from monteplan.config.defaults import default_market
        from monteplan.config.schema import (
            AccountConfig,
            PlanConfig,
            PolicyBundle,
            SimulationConfig,
        )
        from monteplan.core.engine import simulate

        # No returns and no inflation, so every path spends exactly 12 * 4,000 from the
        # traditional account and pays the tax on it.
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
            retirement_age=60,
            end_age=61,
            monthly_spending=4_000,
            accounts=[AccountConfig(account_type="traditional", balance=1e6)],
        )
        result = simulate(
            plan, market, PolicyBundle(tax_model="us_federal"), SimulationConfig(n_paths=10)
        )
        tax = USFederalTaxModel().compute_annual_tax(48_000, 0, "single")
        assert result.terminal_wealth_percentiles["p50"] == pytest.approx(1e6 - 48_000 - tax)
