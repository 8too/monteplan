"""Medicare Part B premiums and income-related surcharges (IRMAA)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from monteplan.io.yaml_loader import load_package_yaml

# Year of the tables used by the engine.
MEDICARE_YEAR = 2026


class MedicarePremiums:
    """The standard Part B premium and the IRMAA surcharges on Parts B and D.

    The surcharges depend on modified adjusted gross income of two years earlier. Amounts
    are those of the table year; the income thresholds of later years are indexed by
    ``inflation_index``, the price level relative to the table year.
    """

    def __init__(self, year: int = MEDICARE_YEAR) -> None:
        data: dict[str, Any] = load_package_yaml(f"healthcare/tables/medicare_{year}.yaml")
        self.part_b_premium: float = data["part_b_premium"]
        self._irmaa: dict[str, list[list[float]]] = data["irmaa"]

    def irmaa(
        self,
        income: NDArray[np.floating[Any]],
        filing_status: str,
        inflation_index: NDArray[np.floating[Any]] | float = 1.0,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Monthly Part B and Part D surcharges per enrollee.

        Args:
            income: (n_paths,) modified adjusted gross income of two years earlier.
            filing_status: ``"single"`` or ``"married_jointly"``.
            inflation_index: Price level relative to the table year, scalar or per path.

        Returns:
            (n_paths,) Part B and (n_paths,) Part D monthly surcharges.
        """
        index = np.asarray(inflation_index, dtype=float)
        part_b: NDArray[np.floating[Any]] = np.zeros(np.shape(income))
        part_d: NDArray[np.floating[Any]] = np.zeros(np.shape(income))
        tiers = self._irmaa[filing_status]
        for i, (threshold, b_surcharge, d_surcharge) in enumerate(tiers):
            # Tiers apply above their threshold; the top one from its threshold on.
            bound = threshold * index
            in_tier = income >= bound if i == len(tiers) - 1 else income > bound
            part_b = np.where(in_tier, b_surcharge, part_b)
            part_d = np.where(in_tier, d_surcharge, part_d)
        return part_b, part_d
