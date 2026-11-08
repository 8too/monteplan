"""Tests for US federal bracket-based tax model."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.config.schema import MarketAssumptions
from monteplan.taxes.us_federal import USFederalTaxModel


@pytest.fixture
def tax_model() -> USFederalTaxModel:
    return USFederalTaxModel(tax_year=2026)


# Taxable income at each bracket threshold and the tax on it, from the tax rate tables of
# IRS Rev. Proc. 2025-32, section 4.01.
IRS_2026_TAX_AT_THRESHOLDS = {
    "single": [
        (12_400, 1_240.0),
        (50_400, 5_800.0),
        (105_700, 17_966.0),
        (201_775, 41_024.0),
        (256_225, 58_448.0),
        (640_600, 192_979.25),
    ],
    "married_jointly": [
        (24_800, 2_480.0),
        (100_800, 11_600.0),
        (211_400, 35_932.0),
        (403_550, 82_048.0),
        (512_450, 116_896.0),
        (768_700, 206_583.50),
    ],
}


class TestOrdinaryIncome:
    def test_default_tax_year_is_2026(self) -> None:
        assert USFederalTaxModel().standard_deduction("single") == 16_100

    def test_standard_deduction_single(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.standard_deduction("single") == 16_100

    def test_standard_deduction_married(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.standard_deduction("married_jointly") == 32_200

    def test_zero_income(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.compute_annual_tax(0, 0, "single") == 0.0

    def test_below_standard_deduction(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.compute_annual_tax(16_000, 0, "single") == 0.0

    @pytest.mark.parametrize(
        ("filing_status", "taxable_income", "irs_tax"),
        [
            (status, income, tax)
            for status, rows in IRS_2026_TAX_AT_THRESHOLDS.items()
            for income, tax in rows
        ],
    )
    def test_matches_irs_tax_rate_tables(
        self,
        tax_model: USFederalTaxModel,
        filing_status: str,
        taxable_income: float,
        irs_tax: float,
    ) -> None:
        gross_income = taxable_income + tax_model.standard_deduction(filing_status)
        tax = tax_model.compute_annual_tax(gross_income, 0, filing_status)
        assert tax == pytest.approx(irs_tax)

    def test_top_bracket(self, tax_model: USFederalTaxModel) -> None:
        # Taxable 1,000,000: 192,979.25 + 37% of (1,000,000 - 640,600)
        tax = tax_model.compute_annual_tax(1_016_100, 0, "single")
        assert tax == pytest.approx(192_979.25 + 0.37 * 359_400)

    def test_single_50k_income(self, tax_model: USFederalTaxModel) -> None:
        # Taxable = 50,000 - 16,100 = 33,900
        # 10% of 12,400 = 1,240 plus 12% of 21,500 = 2,580
        assert tax_model.compute_annual_tax(50_000, 0, "single") == pytest.approx(3_820.0)

    def test_married_150k_income(self, tax_model: USFederalTaxModel) -> None:
        # Taxable = 150,000 - 32,200 = 117,800
        # 11,600 plus 22% of (117,800 - 100,800) = 3,740
        tax = tax_model.compute_annual_tax(150_000, 0, "married_jointly")
        assert tax == pytest.approx(15_340.0)


class TestLongTermGains:
    """Long-term gains are the top layer of taxable income."""

    def test_within_standard_deduction(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.compute_annual_tax(0, 16_100, "single") == 0.0

    def test_unused_deduction_shelters_gains(self, tax_model: USFederalTaxModel) -> None:
        # Taxable gains = 60,000 - 16,100 = 43,900, all in the 0% bracket (up to 49,450)
        assert tax_model.compute_annual_tax(0, 60_000, "single") == 0.0

    def test_partly_unused_deduction(self, tax_model: USFederalTaxModel) -> None:
        # Ordinary 10,000 leaves 6,100 of the deduction; taxable gains = 64,000 - 6,100 =
        # 57,900: 0% up to 49,450, 15% of 8,450 = 1,267.50
        assert tax_model.compute_annual_tax(10_000, 64_000, "single") == pytest.approx(1_267.5)

    def test_gains_only_in_15_percent_bracket(self, tax_model: USFederalTaxModel) -> None:
        # Taxable gains = 100,000 - 16,100 = 83,900: 15% of (83,900 - 49,450) = 5,167.50
        assert tax_model.compute_annual_tax(0, 100_000, "single") == pytest.approx(5_167.5)

    def test_gains_only_in_20_percent_bracket(self, tax_model: USFederalTaxModel) -> None:
        # Taxable gains = 583,900: 15% of (545,500 - 49,450) = 74,407.50
        # plus 20% of (583,900 - 545,500) = 7,680
        tax = tax_model.compute_annual_tax(0, 600_000, "single")
        assert tax == pytest.approx(82_087.5)

    def test_ordinary_income_fills_zero_bracket_first(
        self, tax_model: USFederalTaxModel
    ) -> None:
        # Taxable ordinary = 80,000 - 16,100 = 63,900, above the 0% gains bracket, so all
        # 40,000 of gains are taxed at 15% = 6,000.
        # Ordinary tax: 5,800 + 22% of 13,500 = 8,770
        tax = tax_model.compute_annual_tax(80_000, 40_000, "single")
        assert tax == pytest.approx(8_770.0 + 6_000.0)

    def test_gains_straddle_zero_bracket(self, tax_model: USFederalTaxModel) -> None:
        # Taxable ordinary = 40,000 - 16,100 = 23,900 leaves 25,550 of the 0% bracket;
        # 15% of the remaining 30,000 - 25,550 = 4,450 is 667.50.
        # Ordinary tax: 1,240 + 12% of 11,500 = 2,620
        tax = tax_model.compute_annual_tax(40_000, 30_000, "single")
        assert tax == pytest.approx(2_620.0 + 667.5)

    def test_gains_straddle_20_percent_bracket(self, tax_model: USFederalTaxModel) -> None:
        # Taxable ordinary = 500,000 - 16,100 = 483,900 leaves 61,600 of the 15% bracket;
        # 15% of 61,600 = 9,240 plus 20% of 100,000 - 61,600 = 38,400 is 7,680.
        # Ordinary tax: 58,448 + 35% of (483,900 - 256,225) = 138,134.25
        tax = tax_model.compute_annual_tax(500_000, 100_000, "single")
        assert tax == pytest.approx(138_134.25 + 9_240.0 + 7_680.0)

    def test_married_stacking(self, tax_model: USFederalTaxModel) -> None:
        # Taxable ordinary = 117,800, above the 0% bracket (up to 98,900): 15% of 50,000
        tax = tax_model.compute_annual_tax(150_000, 50_000, "married_jointly")
        assert tax == pytest.approx(15_340.0 + 7_500.0)

    def test_gains_never_lower_the_tax(self, tax_model: USFederalTaxModel) -> None:
        ordinary = np.repeat([0.0, 10_000.0, 40_000.0, 80_000.0, 300_000.0], 5)
        ltcg = np.tile([0.0, 5_000.0, 30_000.0, 100_000.0, 1e6], 5)
        with_gains = tax_model.compute_annual_tax_vectorized(ordinary, ltcg, "single")
        without_gains = tax_model.compute_annual_tax_vectorized(
            ordinary, np.zeros_like(ltcg), "single"
        )
        assert np.all(with_gains >= without_gains)
        # At most the 20% top rate (plus nothing else) on the gains
        assert np.all(with_gains - without_gains <= 0.2 * ltcg + 1e-9)

    def test_vectorized_matches_scalar(self, tax_model: USFederalTaxModel) -> None:
        ordinary = np.array([0.0, 10_000.0, 40_000.0, 80_000.0, 500_000.0])
        ltcg = np.array([60_000.0, 64_000.0, 30_000.0, 40_000.0, 100_000.0])
        vectorized = tax_model.compute_annual_tax_vectorized(ordinary, ltcg, "single")
        scalar = [
            tax_model.compute_annual_tax(o, g, "single") for o, g in zip(ordinary, ltcg)
        ]
        assert vectorized == pytest.approx(scalar)


class TestInflationIndexing:
    """Tax amounts grow with the inflation index."""

    def test_standard_deduction_indexed(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.standard_deduction("single", 1.1) == pytest.approx(17_710.0)

    def test_indexed_ordinary_tax(self, tax_model: USFederalTaxModel) -> None:
        # Deduction 17,710, 10% bracket up to 13,640. Taxable = 50,000 - 17,710 = 32,290:
        # 10% of 13,640 = 1,364 plus 12% of 18,650 = 2,238
        tax = tax_model.compute_annual_tax(50_000, 0, "single", inflation_index=1.1)
        assert tax == pytest.approx(3_602.0)

    def test_indexed_gains_brackets(self, tax_model: USFederalTaxModel) -> None:
        # Deduction 17,710, 0% gains bracket up to 54,395. Taxable gains = 100,000 -
        # 17,710 = 82,290: 15% of (82,290 - 54,395) = 4,184.25
        tax = tax_model.compute_annual_tax(0, 100_000, "single", inflation_index=1.1)
        assert tax == pytest.approx(4_184.25)

    @pytest.mark.parametrize("filing_status", ["single", "married_jointly"])
    def test_real_tax_unchanged_by_inflation(
        self, tax_model: USFederalTaxModel, filing_status: str
    ) -> None:
        """Income that keeps up with inflation pays the same real tax."""
        ordinary = np.repeat([0.0, 30_000.0, 120_000.0, 700_000.0], 4)
        ltcg = np.tile([0.0, 20_000.0, 90_000.0, 650_000.0], 4)
        base = tax_model.compute_annual_tax_vectorized(ordinary, ltcg, filing_status)
        for index in [0.8, 1.37, 2.5]:
            indexed = tax_model.compute_annual_tax_vectorized(
                ordinary * index, ltcg * index, filing_status, index
            )
            assert indexed == pytest.approx(base * index)

    def test_same_income_pays_less_after_inflation(self, tax_model: USFederalTaxModel) -> None:
        nominal = tax_model.compute_annual_tax(100_000, 50_000, "single")
        indexed = tax_model.compute_annual_tax(100_000, 50_000, "single", inflation_index=1.5)
        assert indexed < nominal

    def test_index_per_path(self, tax_model: USFederalTaxModel) -> None:
        ordinary = np.array([50_000.0, 50_000.0, 80_000.0])
        ltcg = np.array([0.0, 100_000.0, 40_000.0])
        index = np.array([1.0, 1.1, 1.6])
        vectorized = tax_model.compute_annual_tax_vectorized(ordinary, ltcg, "single", index)
        scalar = [
            tax_model.compute_annual_tax(o, g, "single", i)
            for o, g, i in zip(ordinary, ltcg, index)
        ]
        assert vectorized == pytest.approx(scalar)

    def test_marginal_rate_indexed(self, tax_model: USFederalTaxModel) -> None:
        assert tax_model.marginal_rate(52_000, "single") == 0.22
        assert tax_model.marginal_rate(52_000, "single", inflation_index=1.1) == 0.12

    def test_niit_not_indexed(self, tax_model: USFederalTaxModel) -> None:
        """The NIIT thresholds are fixed by statute."""
        niit = tax_model.compute_niit_vectorized(
            np.array([190_000.0]), np.array([30_000.0]), "single"
        )
        assert niit[0] == pytest.approx(20_000 * 0.038)


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
        assert tax_model.tax_on_income(50_000) == pytest.approx(3_820.0)


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

    def test_tax_paid_from_traditional_is_next_years_income(self) -> None:
        """Paying the tax from a traditional account is a taxable distribution itself."""
        from monteplan.config.defaults import default_market
        from monteplan.config.schema import (
            AccountConfig,
            PlanConfig,
            PolicyBundle,
            SimulationConfig,
        )
        from monteplan.core.engine import simulate

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
            end_age=62,
            monthly_spending=4_000,
            accounts=[AccountConfig(account_type="traditional", balance=1e6)],
        )
        result = simulate(
            plan, market, PolicyBundle(tax_model="us_federal"), SimulationConfig(n_paths=10)
        )
        tax_model = USFederalTaxModel()
        first_year_tax = tax_model.compute_annual_tax(48_000, 0, "single")
        second_year_tax = tax_model.compute_annual_tax(48_000 + first_year_tax, 0, "single")
        assert result.terminal_wealth_percentiles["p50"] == pytest.approx(
            1e6 - 96_000 - first_year_tax - second_year_tax
        )


class TestEngineInflationIndexing:
    """The engine indexes each year's tax amounts by the inflation since the start."""

    @staticmethod
    def _market(inflation: float) -> MarketAssumptions:
        from monteplan.config.defaults import default_market

        return default_market().model_copy(
            update={
                "expected_annual_returns": [0.0] * 6,
                "annual_volatilities": [0.0] * 6,
                "inflation_mean": inflation,
                "inflation_vol": 0.0,
            }
        )

    def test_second_year_tax_indexed(self) -> None:
        from monteplan.config.schema import (
            AccountConfig,
            PlanConfig,
            PolicyBundle,
            SimulationConfig,
        )
        from monteplan.core.engine import simulate

        plan = PlanConfig(
            current_age=60,
            retirement_age=60,
            end_age=62,
            monthly_spending=6_000,
            accounts=[AccountConfig(account_type="traditional", balance=1e6)],
        )
        result = simulate(
            plan,
            self._market(0.10),
            PolicyBundle(tax_model="us_federal"),
            SimulationConfig(n_paths=5, store_paths=True),
        )
        assert result.spending_history is not None and result.inflation_rates is not None
        first_year_spending = result.spending_history[0, :12].sum()
        second_year_spending = result.spending_history[0, 12:24].sum()
        index = np.prod(1.0 + result.inflation_rates[0, :12])
        assert index > 1.05

        tax_model = USFederalTaxModel()
        first_year_tax = tax_model.compute_annual_tax(first_year_spending, 0, "single")
        second_year_income = second_year_spending + first_year_tax
        second_year_tax = tax_model.compute_annual_tax(
            second_year_income, 0, "single", inflation_index=index
        )
        unindexed_tax = tax_model.compute_annual_tax(second_year_income, 0, "single")
        assert unindexed_tax - second_year_tax > 500
        assert result.terminal_wealth_percentiles["p50"] == pytest.approx(
            1e6 - first_year_spending - second_year_spending - first_year_tax - second_year_tax
        )

    def test_fill_bracket_roth_conversion_ceiling_indexed(self) -> None:
        from monteplan.config.schema import (
            AccountConfig,
            PlanConfig,
            PolicyBundle,
            RothConversionConfig,
            SimulationConfig,
        )
        from monteplan.core.engine import simulate

        plan = PlanConfig(
            current_age=60,
            retirement_age=60,
            end_age=62,
            monthly_spending=0,
            accounts=[
                AccountConfig(account_type="traditional", balance=1e6),
                AccountConfig(account_type="roth", balance=0.0),
            ],
        )
        policies = PolicyBundle(
            tax_model="us_federal",
            roth_conversion=RothConversionConfig(
                enabled=True,
                strategy="fill_bracket",
                fill_to_bracket_top=0.12,
                start_age=60,
                end_age=62,
            ),
        )
        result = simulate(
            plan, self._market(0.10), policies, SimulationConfig(n_paths=5, store_paths=True)
        )
        assert result.inflation_rates is not None
        index = np.prod(1.0 + result.inflation_rates[0, :12])
        # Each year converts up to the top of the 12% bracket: 50,400 + 16,100 = 66,500 in
        # the first year and that times the index in the second, whose income then also
        # includes the first year's tax paid from the traditional account. The tax on
        # income at the top of the 12% bracket is 5,800, indexed.
        assert result.terminal_wealth_percentiles["p50"] == pytest.approx(
            1e6 - 5_800 - 5_800 * index
        )
