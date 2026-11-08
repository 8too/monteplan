"""Healthcare costs of all paths through retirement."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from monteplan.config.schema import HealthcareCost
from monteplan.healthcare.aca import PremiumTaxCredit
from monteplan.healthcare.medicare import MedicarePremiums


class HealthcareModel:
    """Monthly healthcare costs and the year-end premium tax credit, per path.

    Before the Medicare age the household pays the full premium of its ACA plan each month
    and gets the premium tax credit for the year's income at year end. From the Medicare age
    each enrollee pays the Part B premium and the IRMAA surcharges set by the income of two
    years earlier. Income before the simulation start is unknown, so the first two years
    have no surcharges.

    Premiums, surcharges and out-of-pocket costs grow at the configured rate from the
    simulation start; the income thresholds are indexed by simulated inflation.
    """

    def __init__(self, config: HealthcareCost, filing_status: str, n_paths: int) -> None:
        self._config = config
        self._filing_status = filing_status
        self._monthly_growth = (1.0 + config.growth_rate) ** (1.0 / 12.0) - 1.0
        self._credit = (
            None if config.aca_subsidy == "none" else PremiumTaxCredit(config.aca_subsidy)
        )
        self._medicare = MedicarePremiums()
        # ACA premiums of the year so far, of the plan bought and of the benchmark plan.
        self._plan_premiums = np.zeros(n_paths)
        self._benchmark_premiums = np.zeros(n_paths)
        # The year's monthly IRMAA surcharges per enrollee, in table-year dollars.
        self._irmaa_part_b = np.zeros(n_paths)
        self._irmaa_part_d = np.zeros(n_paths)
        self._last_year_income: NDArray[np.floating[Any]] | None = None

    def monthly_cost(
        self, step: int, age: float, paying: NDArray[np.bool_]
    ) -> NDArray[np.floating[Any]]:
        """The month's healthcare costs per path, before any premium tax credit.

        Args:
            step: Month since the simulation start.
            age: Age in the month.
            paying: (n_paths,) paths that pay the costs; only their ACA premiums count
                toward the year's premium tax credit.
        """
        config = self._config
        growth = (1.0 + self._monthly_growth) ** step
        out_of_pocket = config.monthly_out_of_pocket * growth
        if age < config.medicare_age:
            plan = config.monthly_premium * growth
            self._plan_premiums += np.where(paying, plan, 0.0)
            self._benchmark_premiums += np.where(
                paying, config.monthly_benchmark_premium * growth, 0.0
            )
            return np.full(paying.shape, plan + out_of_pocket)
        per_enrollee = (
            self._medicare.part_b_premium + self._irmaa_part_b + self._irmaa_part_d
        ) * growth
        result: NDArray[np.floating[Any]] = (
            config.household_size * per_enrollee
            + config.monthly_medicare_supplement * growth
            + out_of_pocket
        )
        return result

    def close_year(
        self,
        income: NDArray[np.floating[Any]],
        inflation_index: NDArray[np.floating[Any]],
        next_inflation_index: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Ends a year: returns its premium tax credit and sets next year's surcharges.

        Args:
            income: (n_paths,) modified adjusted gross income of the year.
            inflation_index: (n_paths,) price level of the year relative to the start.
            next_inflation_index: (n_paths,) price level of the next year.

        Returns:
            (n_paths,) premium tax credit for the year's ACA premiums.
        """
        credit: NDArray[np.floating[Any]] = np.zeros_like(income)
        if self._credit is not None and self._plan_premiums.any():
            credit = self._credit.credit(
                income,
                self._plan_premiums,
                self._benchmark_premiums,
                self._config.household_size,
                inflation_index,
            )
        self._plan_premiums = np.zeros_like(self._plan_premiums)
        self._benchmark_premiums = np.zeros_like(self._benchmark_premiums)
        # Next year's surcharges depend on the income two years before it: this year's
        # surcharges were set by the year before, so next year's by the last one.
        if self._last_year_income is not None:
            self._irmaa_part_b, self._irmaa_part_d = self._medicare.irmaa(
                self._last_year_income, self._filing_status, next_inflation_index
            )
        self._last_year_income = income.copy()
        return credit
