"""US Federal bracket-based income tax model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from monteplan.io.yaml_loader import load_package_yaml

# Tax table year used by the engine.
TAX_YEAR = 2026


class USFederalTaxModel:
    """US Federal income tax with progressive brackets.

    Loads bracket tables from YAML data files. Supports ordinary income brackets and
    long-term capital gains rates.

    The tables hold the amounts of their tax year. Later years are modeled by passing
    ``inflation_index``, the price level relative to the table year (a scalar or one value
    per path), which scales the standard deduction and every bracket threshold the way the
    IRS indexes them each year. The NIIT thresholds are fixed by statute and not indexed.
    """

    def __init__(self, tax_year: int = TAX_YEAR) -> None:
        data: dict[str, Any] = load_package_yaml(f"taxes/tables/us_federal_{tax_year}.yaml")
        self._standard_deduction: dict[str, float] = data["standard_deduction"]
        self._ordinary_brackets: dict[str, list[list[Any]]] = data["ordinary_brackets"]
        self._ltcg_brackets: dict[str, list[list[Any]]] = data["ltcg_brackets"]

    @staticmethod
    def _apply_brackets(
        taxable_income: NDArray[np.floating[Any]],
        brackets: list[list[Any]],
        inflation_index: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Progressive tax on taxable income, with bracket bounds scaled by the index."""
        tax: NDArray[np.floating[Any]] = np.zeros(
            np.broadcast(taxable_income, inflation_index).shape
        )
        prev_bound: NDArray[np.floating[Any]] | float = 0.0
        for upper_bound, rate in brackets:
            bound = np.inf if upper_bound is None else upper_bound * inflation_index
            taxable_in_bracket = np.minimum(taxable_income, bound) - prev_bound
            tax += np.maximum(taxable_in_bracket, 0.0) * rate
            prev_bound = bound
        return tax

    def standard_deduction(self, filing_status: str, inflation_index: float = 1.0) -> float:
        """Return the standard deduction for a filing status, indexed by inflation."""
        return self._standard_deduction[filing_status] * inflation_index

    def tax_on_income(self, gross_income: float) -> float:
        """Compute tax on ordinary income (single filer, after standard deduction)."""
        return self.compute_annual_tax(gross_income, 0.0, "single")

    def tax_rate_traditional(self) -> float:
        """Approximate effective rate for traditional withdrawals.

        Returns a moderate estimate (22% bracket) for use in gross-up
        calculations during the monthly loop.
        """
        return 0.22

    def compute_annual_tax(
        self,
        ordinary_income: float,
        ltcg: float,
        filing_status: str,
        inflation_index: float = 1.0,
    ) -> float:
        """Compute total annual federal tax; see ``compute_annual_tax_vectorized()``."""
        tax = self.compute_annual_tax_vectorized(
            np.array([ordinary_income], dtype=float),
            np.array([ltcg], dtype=float),
            filing_status,
            inflation_index,
        )
        return float(tax[0])

    def compute_annual_tax_vectorized(
        self,
        ordinary_income: NDArray[np.floating[Any]],
        ltcg: NDArray[np.floating[Any]],
        filing_status: str,
        inflation_index: NDArray[np.floating[Any]] | float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Vectorized annual tax across all paths.

        Ordinary income is reduced by the standard deduction and taxed at the ordinary
        rates. Long-term gains are reduced by any standard deduction ordinary income left
        unused, and are then the top layer of taxable income: each gains bracket applies to
        the part of the gains whose position above the taxable ordinary income falls into
        it. Ordinary income thus fills the 0% gains bracket first.

        Args:
            ordinary_income: (n_paths,) ordinary income per path, including net
                short-term gains.
            ltcg: (n_paths,) net long-term capital gains per path (non-negative).
            filing_status: Filing status key.
            inflation_index: Price level relative to the tax table year, as a scalar or
                (n_paths,) array.

        Returns:
            (n_paths,) total tax liability per path.
        """
        index = np.asarray(inflation_index, dtype=float)
        deduction = self._standard_deduction[filing_status] * index
        taxable_ordinary = np.maximum(ordinary_income - deduction, 0.0)
        unused_deduction = np.maximum(deduction - ordinary_income, 0.0)
        taxable_ltcg = np.maximum(ltcg - unused_deduction, 0.0)

        ordinary_tax = self._apply_brackets(
            taxable_ordinary, self._ordinary_brackets[filing_status], index
        )
        # Tax on the slice of taxable income from taxable_ordinary up to taxable_ordinary +
        # taxable_ltcg, at the gains rates.
        ltcg_brackets = self._ltcg_brackets[filing_status]
        ltcg_tax = self._apply_brackets(
            taxable_ordinary + taxable_ltcg, ltcg_brackets, index
        ) - self._apply_brackets(taxable_ordinary, ltcg_brackets, index)
        result: NDArray[np.floating[Any]] = ordinary_tax + ltcg_tax
        return result

    _NIIT_RATE: float = 0.038
    _NIIT_THRESHOLDS: dict[str, float] = {
        "single": 200_000,
        "married_jointly": 250_000,
    }

    def compute_niit_vectorized(
        self,
        ordinary_income: NDArray[np.floating[Any]],
        ltcg: NDArray[np.floating[Any]],
        filing_status: str,
    ) -> NDArray[np.floating[Any]]:
        """Compute Net Investment Income Tax (3.8% surtax).

        NIIT applies to the lesser of net investment income (here approximated
        as LTCG) or the excess of MAGI over the threshold. The thresholds are not
        indexed for inflation.

        Args:
            ordinary_income: (n_paths,) ordinary income per path.
            ltcg: (n_paths,) long-term capital gains per path.
            filing_status: Filing status key.

        Returns:
            (n_paths,) NIIT liability per path.
        """
        threshold = self._NIIT_THRESHOLDS[filing_status]
        magi = ordinary_income + ltcg
        excess = np.maximum(magi - threshold, 0.0)
        result: NDArray[np.floating[Any]] = np.minimum(ltcg, excess) * self._NIIT_RATE
        return result

    def bracket_ceiling(self, target_rate: float, filing_status: str) -> float:
        """Return the gross income ceiling for a target marginal rate.

        Returns the maximum gross income (bracket upper bound + standard
        deduction) at which the marginal rate is still at or below
        ``target_rate``, in the amounts of the table year. Since both parts are
        indexed, the ceiling of a later year is this times its inflation index.
        Used by fill_bracket Roth conversion strategy.

        Args:
            target_rate: Target marginal tax rate (e.g. 0.22 for 22% bracket).
            filing_status: Filing status key.

        Returns:
            Maximum gross income before exceeding the target bracket.
        """
        std_ded = self._standard_deduction[filing_status]
        brackets = self._ordinary_brackets[filing_status]
        ceiling = 0.0
        for upper_bound, rate in brackets:
            if rate > target_rate:
                break
            ceiling = 10_000_000.0 if upper_bound is None else float(upper_bound)
        return ceiling + std_ded

    def marginal_rate(
        self, taxable_income: float, filing_status: str, inflation_index: float = 1.0
    ) -> float:
        """Return the marginal ordinary income tax rate at given taxable income."""
        brackets = self._ordinary_brackets[filing_status]
        for upper_bound, rate in brackets:
            if upper_bound is None or taxable_income <= upper_bound * inflation_index:
                return float(rate)
        # Should not reach here; return top rate
        return float(brackets[-1][1])
