"""Tests for Medicare Part B premiums and IRMAA surcharges."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.healthcare.medicare import MedicarePremiums


@pytest.fixture
def medicare() -> MedicarePremiums:
    return MedicarePremiums()


def surcharges(
    medicare: MedicarePremiums, income: float, filing_status: str, index: float = 1.0
) -> tuple[float, float]:
    part_b, part_d = medicare.irmaa(np.array([income]), filing_status, index)
    return float(part_b[0]), float(part_d[0])


def test_standard_part_b_premium(medicare: MedicarePremiums) -> None:
    assert medicare.part_b_premium == 202.90


@pytest.mark.parametrize(
    ("income", "part_b", "part_d"),
    [
        (0, 0.0, 0.0),
        (109_000, 0.0, 0.0),
        (109_001, 81.20, 14.50),
        (137_000, 81.20, 14.50),
        (137_001, 202.90, 37.50),
        (171_001, 324.60, 60.40),
        (205_001, 446.30, 83.30),
        (499_999, 446.30, 83.30),
        (500_000, 487.00, 91.00),
    ],
)
def test_single_tiers(
    medicare: MedicarePremiums, income: float, part_b: float, part_d: float
) -> None:
    assert surcharges(medicare, income, "single") == pytest.approx((part_b, part_d))


@pytest.mark.parametrize(
    ("income", "part_b", "part_d"),
    [
        (218_000, 0.0, 0.0),
        (218_001, 81.20, 14.50),
        (342_001, 324.60, 60.40),
        (749_999, 446.30, 83.30),
        (750_000, 487.00, 91.00),
    ],
)
def test_married_jointly_tiers(
    medicare: MedicarePremiums, income: float, part_b: float, part_d: float
) -> None:
    assert surcharges(medicare, income, "married_jointly") == pytest.approx((part_b, part_d))


def test_thresholds_indexed(medicare: MedicarePremiums) -> None:
    # 115,000 is above 109,000 but below 109,000 * 1.1 = 119,900
    assert surcharges(medicare, 115_000, "single") == pytest.approx((81.20, 14.50))
    assert surcharges(medicare, 115_000, "single", 1.1) == (0.0, 0.0)


def test_per_path_index(medicare: MedicarePremiums) -> None:
    part_b, _ = medicare.irmaa(np.array([115_000.0, 115_000.0]), "single", np.array([1.0, 1.1]))
    assert part_b == pytest.approx([81.20, 0.0])
