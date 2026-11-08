"""Tests for the ACA premium tax credit."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.healthcare.aca import PremiumTaxCredit

# 2026 coverage uses the 2025 poverty line: 15,650 for one person, plus 5,500 per person.
FPL_1 = 15_650.0


@pytest.fixture
def original() -> PremiumTaxCredit:
    return PremiumTaxCredit("original")


@pytest.fixture
def enhanced() -> PremiumTaxCredit:
    return PremiumTaxCredit("enhanced")


def credit(
    model: PremiumTaxCredit,
    income: float,
    plan: float,
    benchmark: float,
    household_size: int = 1,
    inflation_index: float = 1.0,
) -> float:
    return float(
        model.credit(np.array([income]), plan, benchmark, household_size, inflation_index)[0]
    )


class TestPovertyLine:
    def test_household_sizes(self, original: PremiumTaxCredit) -> None:
        assert original.poverty_line(1) == FPL_1
        assert original.poverty_line(2) == 21_150

    def test_indexed(self, original: PremiumTaxCredit) -> None:
        assert original.poverty_line(1, 1.1) == pytest.approx(17_215)


class TestApplicablePercentage:
    @pytest.mark.parametrize(
        ("poverty_ratio", "percentage"),
        [
            (0.0, 0.0210),
            (1.0, 0.0210),
            (1.33, 0.0314),
            # Halfway through the 133-150% band: halfway from 3.14% to 4.19%
            (1.415, (0.0314 + 0.0419) / 2),
            (1.5, 0.0419),
            (2.25, (0.0660 + 0.0844) / 2),
            (2.5, 0.0844),
            (2.99, 0.0844 + (0.0996 - 0.0844) * 0.98),
            (3.5, 0.0996),
            (4.0, 0.0996),
        ],
    )
    def test_original_table(
        self, original: PremiumTaxCredit, poverty_ratio: float, percentage: float
    ) -> None:
        result = original.applicable_percentage(np.array([poverty_ratio]))
        assert result[0] == pytest.approx(percentage)

    def test_original_table_ends_at_400_percent(self, original: PremiumTaxCredit) -> None:
        assert np.isnan(original.applicable_percentage(np.array([4.001]))[0])

    @pytest.mark.parametrize(
        ("poverty_ratio", "percentage"),
        [(1.0, 0.0), (1.75, 0.01), (2.0, 0.02), (3.5, 0.0725), (4.0, 0.085), (10.0, 0.085)],
    )
    def test_enhanced_table(
        self, enhanced: PremiumTaxCredit, poverty_ratio: float, percentage: float
    ) -> None:
        result = enhanced.applicable_percentage(np.array([poverty_ratio]))
        assert result[0] == pytest.approx(percentage)


class TestCredit:
    def test_benchmark_minus_contribution(self, original: PremiumTaxCredit) -> None:
        # 200% FPL = 31,300: contribution 6.60% = 2,065.80; credit 12,000 - 2,065.80
        result = credit(original, 2 * FPL_1, plan=14_400, benchmark=12_000)
        assert result == pytest.approx(9_934.20)
        # The net premium is plan - benchmark + contribution
        assert 14_400 - result == pytest.approx(14_400 - 12_000 + 2_065.80)

    def test_capped_at_plan_premium(self, original: PremiumTaxCredit) -> None:
        assert credit(original, 2 * FPL_1, plan=6_000, benchmark=12_000) == pytest.approx(6_000)

    def test_none_when_contribution_exceeds_benchmark(self, original: PremiumTaxCredit) -> None:
        # 400% FPL = 62,600: contribution 9.96% = 6,234.96 > 6,000
        assert credit(original, 4 * FPL_1, plan=8_000, benchmark=6_000) == 0.0

    def test_cliff_above_400_percent(self, original: PremiumTaxCredit) -> None:
        at_limit = credit(original, 4 * FPL_1, plan=24_000, benchmark=20_000)
        assert at_limit == pytest.approx(20_000 - 6_234.96)
        assert credit(original, 4 * FPL_1 + 1, plan=24_000, benchmark=20_000) == 0.0

    def test_enhanced_has_no_cliff(self, enhanced: PremiumTaxCredit) -> None:
        # 8.5% of 100,000 = 8,500
        result = credit(enhanced, 100_000, plan=24_000, benchmark=20_000)
        assert result == pytest.approx(11_500)

    def test_zero_income(self, original: PremiumTaxCredit) -> None:
        assert credit(original, 0, plan=14_400, benchmark=12_000) == pytest.approx(12_000)

    def test_household_of_two(self, original: PremiumTaxCredit) -> None:
        # 400% of 21,150 = 84,600 is still eligible: 9.96% = 8,426.16
        result = credit(original, 84_600, plan=30_000, benchmark=25_000, household_size=2)
        assert result == pytest.approx(25_000 - 8_426.16)

    def test_poverty_line_indexed(self, original: PremiumTaxCredit) -> None:
        # 68,000 is above 400% of 15,650, but 395% of 17,215: 9.96% = 6,772.80
        assert credit(original, 68_000, plan=24_000, benchmark=20_000) == 0.0
        result = credit(original, 68_000, plan=24_000, benchmark=20_000, inflation_index=1.1)
        assert result == pytest.approx(20_000 - 6_772.80)

    def test_vectorized_per_path(self, original: PremiumTaxCredit) -> None:
        income = np.array([0.0, 20_000.0, 31_300.0, 62_600.0, 70_000.0])
        plan = np.array([14_400.0, 14_400.0, 6_000.0, 24_000.0, 24_000.0])
        benchmark = np.array([12_000.0, 12_000.0, 12_000.0, 20_000.0, 20_000.0])
        index = np.array([1.0, 1.0, 1.0, 1.0, 1.2])
        result = original.credit(income, plan, benchmark, 1, index)
        expected = [
            credit(original, i, p, b, 1, x) for i, p, b, x in zip(income, plan, benchmark, index)
        ]
        assert result == pytest.approx(expected)
