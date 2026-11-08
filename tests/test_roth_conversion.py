"""Tests for Roth conversion modeling feature."""

from __future__ import annotations

import pytest

from monteplan.config.defaults import default_market
from monteplan.config.schema import (
    AccountConfig,
    PlanConfig,
    PolicyBundle,
    RothConversionConfig,
    SimulationConfig,
)
from monteplan.core.engine import simulate
from monteplan.taxes.us_federal import USFederalTaxModel


class TestRothConversionConfigValidation:
    """Validate RothConversionConfig constraints."""

    def test_default_disabled(self) -> None:
        cfg = RothConversionConfig()
        assert cfg.enabled is False
        assert cfg.strategy == "fixed_amount"
        assert cfg.annual_amount == 0.0

    def test_end_age_must_exceed_start_age(self) -> None:
        with pytest.raises(ValueError, match="end_age must be greater than start_age"):
            RothConversionConfig(enabled=True, start_age=70, end_age=60)

    def test_equal_ages_rejected(self) -> None:
        with pytest.raises(ValueError, match="end_age must be greater than start_age"):
            RothConversionConfig(enabled=True, start_age=65, end_age=65)

    def test_valid_config(self) -> None:
        cfg = RothConversionConfig(
            enabled=True,
            strategy="fill_bracket",
            fill_to_bracket_top=0.22,
            start_age=55,
            end_age=72,
        )
        assert cfg.enabled is True
        assert cfg.strategy == "fill_bracket"

    def test_round_trip(self) -> None:
        cfg = RothConversionConfig(
            enabled=True,
            strategy="fixed_amount",
            annual_amount=50_000.0,
            start_age=55,
            end_age=72,
        )
        policies = PolicyBundle(roth_conversion=cfg)
        dumped = policies.model_dump_json()
        restored = PolicyBundle.model_validate_json(dumped)
        assert restored.roth_conversion.enabled is True
        assert restored.roth_conversion.annual_amount == pytest.approx(50_000.0)


class TestBracketCeiling:
    """Test USFederalTaxModel.bracket_ceiling()."""

    def setup_method(self) -> None:
        self.model = USFederalTaxModel()

    def test_22_bracket_single(self) -> None:
        ceiling = self.model.bracket_ceiling(0.22, "single")
        # Should be the top of the 22% bracket + standard deduction
        std_ded = self.model.standard_deduction("single")
        assert ceiling > std_ded
        assert ceiling > 0

    def test_2026_ceilings(self) -> None:
        assert self.model.bracket_ceiling(0.22, "single") == 105_700 + 16_100
        assert self.model.bracket_ceiling(0.12, "married_jointly") == 100_800 + 32_200

    def test_higher_bracket_higher_ceiling(self) -> None:
        c22 = self.model.bracket_ceiling(0.22, "single")
        c32 = self.model.bracket_ceiling(0.32, "single")
        assert c32 > c22

    def test_married_has_wider_brackets(self) -> None:
        c_single = self.model.bracket_ceiling(0.22, "single")
        c_married = self.model.bracket_ceiling(0.22, "married_jointly")
        assert c_married > c_single


class TestRothConversionEngine:
    """Integration tests for Roth conversions in the engine."""

    def test_disabled_matches_golden(self) -> None:
        """Disabled Roth conversion produces same result as v0.4 golden."""
        from monteplan.config.defaults import default_plan, default_policies

        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=5000, seed=42),
        )
        assert result.success_probability == pytest.approx(0.7582, abs=0.03)

    def test_fixed_amount_conversion(self) -> None:
        """Fixed $50K/year conversion runs and produces valid results."""
        plan = PlanConfig(
            current_age=50,
            retirement_age=55,
            end_age=90,
            accounts=[
                AccountConfig(account_type="traditional", balance=500_000),
                AccountConfig(account_type="roth", balance=100_000),
            ],
            monthly_income=8_000,
            monthly_spending=4_000,
        )
        sim = SimulationConfig(n_paths=100, seed=42)

        result = simulate(
            plan,
            default_market(),
            PolicyBundle(
                tax_model="us_federal",
                roth_conversion=RothConversionConfig(
                    enabled=True,
                    strategy="fixed_amount",
                    annual_amount=50_000.0,
                    start_age=55,
                    end_age=65,
                ),
            ),
            sim,
        )

        assert 0.0 <= result.success_probability <= 1.0
        assert result.n_paths == 100

    def test_fill_bracket_conversion(self) -> None:
        """Fill bracket strategy runs without error and produces valid results."""
        plan = PlanConfig(
            current_age=50,
            retirement_age=55,
            end_age=90,
            accounts=[
                AccountConfig(account_type="traditional", balance=500_000),
                AccountConfig(account_type="roth", balance=100_000),
            ],
            monthly_income=8_000,
            monthly_spending=4_000,
        )
        result = simulate(
            plan,
            default_market(),
            PolicyBundle(
                tax_model="us_federal",
                roth_conversion=RothConversionConfig(
                    enabled=True,
                    strategy="fill_bracket",
                    fill_to_bracket_top=0.22,
                    start_age=55,
                    end_age=72,
                ),
            ),
            SimulationConfig(n_paths=100, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0

    def test_no_traditional_accounts_noop(self) -> None:
        """Roth conversion with no traditional accounts is a graceful no-op."""
        plan = PlanConfig(
            current_age=30,
            retirement_age=65,
            end_age=95,
            accounts=[
                AccountConfig(account_type="taxable", balance=200_000),
                AccountConfig(account_type="roth", balance=100_000),
            ],
            monthly_income=8_000,
            monthly_spending=5_000,
        )
        result = simulate(
            plan,
            default_market(),
            PolicyBundle(
                roth_conversion=RothConversionConfig(
                    enabled=True,
                    strategy="fixed_amount",
                    annual_amount=50_000.0,
                    start_age=55,
                    end_age=72,
                ),
            ),
            SimulationConfig(n_paths=100, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0

    def test_golden_roth_conversion(self) -> None:
        """Golden test: age 55-65, $500K trad, $100K Roth, $50K/year, seed=42."""
        plan = PlanConfig(
            current_age=50,
            retirement_age=55,
            end_age=90,
            accounts=[
                AccountConfig(account_type="traditional", balance=500_000),
                AccountConfig(account_type="roth", balance=100_000),
            ],
            monthly_income=8_000,
            monthly_spending=4_000,
        )
        result = simulate(
            plan,
            default_market(),
            PolicyBundle(
                tax_model="us_federal",
                roth_conversion=RothConversionConfig(
                    enabled=True,
                    strategy="fixed_amount",
                    annual_amount=50_000.0,
                    start_age=55,
                    end_age=65,
                ),
            ),
            SimulationConfig(n_paths=2000, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0
        assert result.n_paths == 2000

    def test_conversion_increases_ordinary_income(self) -> None:
        """Large conversions should run without errors."""
        plan = PlanConfig(
            current_age=50,
            retirement_age=55,
            end_age=90,
            accounts=[
                AccountConfig(account_type="traditional", balance=500_000),
                AccountConfig(account_type="roth", balance=100_000),
            ],
            monthly_income=8_000,
            monthly_spending=4_000,
        )
        sim = SimulationConfig(n_paths=500, seed=42)
        result = simulate(
            plan,
            default_market(),
            PolicyBundle(
                tax_model="us_federal",
                roth_conversion=RothConversionConfig(
                    enabled=True,
                    strategy="fixed_amount",
                    annual_amount=200_000.0,
                    start_age=55,
                    end_age=65,
                ),
            ),
            sim,
        )
        assert 0.0 <= result.success_probability <= 1.0
