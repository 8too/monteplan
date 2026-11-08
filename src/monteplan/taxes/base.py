"""Base protocol for tax models."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class TaxModel(Protocol):
    """Protocol for tax computation."""

    def tax_on_income(self, gross_income: float) -> float:
        """Compute tax on earned income."""
        ...

    def tax_rate_traditional(self) -> float:
        """Effective tax rate on traditional account withdrawals."""
        ...

    def compute_annual_tax(
        self,
        ordinary_income: float,
        ltcg: float,
        filing_status: str,
        inflation_index: float = 1.0,
    ) -> float:
        """Compute total annual federal tax.

        Args:
            ordinary_income: Gross ordinary income (before standard deduction).
            ltcg: Long-term capital gains.
            filing_status: ``"single"`` or ``"married_jointly"``.
            inflation_index: Price level relative to the tax table year, by which
                bracket amounts are indexed.

        Returns:
            Total tax liability.
        """
        ...

    def compute_annual_tax_vectorized(
        self,
        ordinary_income: NDArray[np.floating[Any]],
        ltcg: NDArray[np.floating[Any]],
        filing_status: str,
        inflation_index: NDArray[np.floating[Any]] | float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Vectorized annual tax across all paths.

        Args:
            ordinary_income: (n_paths,) ordinary income per path.
            ltcg: (n_paths,) long-term capital gains per path.
            filing_status: ``"single"`` or ``"married_jointly"``.
            inflation_index: Price level relative to the tax table year, as a scalar or
                (n_paths,) array.

        Returns:
            (n_paths,) total tax liability per path.
        """
        ...

    def marginal_rate(
        self, taxable_income: float, filing_status: str, inflation_index: float = 1.0
    ) -> float:
        """Marginal ordinary income tax rate at given taxable income."""
        ...
