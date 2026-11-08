"""Simple flat-rate tax model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class FlatTaxModel:
    """Flat effective tax rate applied to traditional withdrawals and income.

    A flat rate has no amounts to index, so ``inflation_index`` is ignored.
    """

    def __init__(self, rate: float = 0.22) -> None:
        self._rate = rate

    def tax_on_income(self, gross_income: float) -> float:
        """Compute tax on earned income."""
        return gross_income * self._rate

    def tax_rate_traditional(self) -> float:
        """Effective tax rate on traditional account withdrawals."""
        return self._rate

    def compute_annual_tax(
        self,
        ordinary_income: float,
        ltcg: float,
        filing_status: str,
        inflation_index: float = 1.0,
    ) -> float:
        """Compute total annual tax at flat rate."""
        return (ordinary_income + ltcg) * self._rate

    def compute_annual_tax_vectorized(
        self,
        ordinary_income: NDArray[np.floating[Any]],
        ltcg: NDArray[np.floating[Any]],
        filing_status: str,
        inflation_index: NDArray[np.floating[Any]] | float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Vectorized annual tax at flat rate across all paths."""
        result: NDArray[np.floating[Any]] = (ordinary_income + ltcg) * self._rate
        return result

    def marginal_rate(
        self, taxable_income: float, filing_status: str, inflation_index: float = 1.0
    ) -> float:
        """Marginal rate is always the flat rate."""
        return self._rate
