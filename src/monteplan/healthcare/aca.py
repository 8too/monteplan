"""ACA marketplace premium tax credit (IRC §36B)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from monteplan.io.yaml_loader import load_package_yaml

# Coverage year of the tables used by the engine.
ACA_YEAR = 2026

SubsidyTable = Literal["original", "enhanced"]


class PremiumTaxCredit:
    """The premium tax credit for marketplace coverage, from the year's household income.

    The credit is the benchmark (second-lowest-cost silver) plan premium minus the expected
    contribution, the applicable percentage of household income, but no more than the
    premium of the plan actually bought and no less than zero:

        credit = min(plan premium, max(0, benchmark premium - applicable % * income))

    The applicable percentage depends on income as a multiple of the poverty line. Under
    the original table, income above 400% of the poverty line gets no credit. Since 2026,
    the credit is reconciled in full at tax time, so computing it from the actual annual
    income gives the credit a household keeps.

    The poverty line is indexed by ``inflation_index``, the price level relative to the
    table year; the applicable percentages are not. Medicaid eligibility at low incomes is
    not modeled: incomes below the poverty line get the lowest applicable percentage.
    """

    def __init__(self, table: SubsidyTable = "original", year: int = ACA_YEAR) -> None:
        data: dict[str, Any] = load_package_yaml(f"healthcare/tables/aca_{year}.yaml")
        self._first_person: float = data["poverty_line"]["first_person"]
        self._additional_person: float = data["poverty_line"]["additional_person"]
        self._bands: list[list[Any]] = data["applicable_percentage"][table]

    def poverty_line(self, household_size: int, inflation_index: float = 1.0) -> float:
        """Annual poverty line for a household, indexed by inflation."""
        return (
            self._first_person + self._additional_person * (household_size - 1)
        ) * inflation_index

    def applicable_percentage(
        self, poverty_ratio: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Applicable percentage by income as a multiple of the poverty line; NaN if none."""
        result: NDArray[np.floating[Any]] = np.full(np.shape(poverty_ratio), np.nan)
        for i, (low, high, initial, final) in enumerate(self._bands):
            is_last = i == len(self._bands) - 1
            if high is None:
                in_band = poverty_ratio >= low
                percentage = np.full(np.shape(poverty_ratio), float(initial))
            else:
                # Bands include their lower bound; the last one also its upper bound.
                below_high = poverty_ratio <= high if is_last else poverty_ratio < high
                in_band = (poverty_ratio >= low) & below_high
                percentage = initial + (final - initial) * (poverty_ratio - low) / (high - low)
            result = np.where(in_band, percentage, result)
        # Below the first band (zero or negative income): the lowest percentage.
        result = np.where(poverty_ratio < self._bands[0][0], self._bands[0][2], result)
        return result

    def credit(
        self,
        household_income: NDArray[np.floating[Any]],
        plan_premium: NDArray[np.floating[Any]] | float,
        benchmark_premium: NDArray[np.floating[Any]] | float,
        household_size: int,
        inflation_index: NDArray[np.floating[Any]] | float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Annual premium tax credit per path.

        Args:
            household_income: (n_paths,) modified adjusted gross income of the year.
            plan_premium: Premiums of the plan bought in the year, per path or scalar.
            benchmark_premium: Premiums of the benchmark plan for the same months.
            household_size: Number of people in the tax household.
            inflation_index: Price level relative to the table year, scalar or per path.

        Returns:
            (n_paths,) credit, between zero and the plan premium.
        """
        poverty_line = self.poverty_line(household_size) * np.asarray(inflation_index, dtype=float)
        percentage = self.applicable_percentage(household_income / poverty_line)
        eligible = ~np.isnan(percentage)
        contribution = np.where(eligible, percentage, 0.0) * np.maximum(household_income, 0.0)
        credit = np.minimum(plan_premium, np.maximum(benchmark_premium - contribution, 0.0))
        result: NDArray[np.floating[Any]] = np.where(eligible, credit, 0.0)
        return result
