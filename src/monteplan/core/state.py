"""Mutable simulation state passed through each time step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationState:
    """Mutable state for all paths at a given time step.

    Uses plain numpy arrays for performance. NOT Pydantic — this is mutated
    every step in the hot loop.

    Attributes:
        positions: (n_paths, n_accounts, n_assets) dollar holdings per asset
            within each account.
        cumulative_inflation: (n_paths,) cumulative inflation factor (starts at 1.0).
        is_depleted: (n_paths,) boolean mask for paths with zero total wealth.
        step: Current time step index.
        n_paths: Number of simulation paths.
        n_accounts: Number of accounts.
        n_assets: Number of asset classes.
        account_types: List of account type strings.
    """

    positions: NDArray[np.floating[Any]]
    cumulative_inflation: NDArray[np.floating[Any]]
    is_depleted: NDArray[np.bool_]
    step: int
    n_paths: int
    n_accounts: int
    n_assets: int
    account_types: list[str] = field(default_factory=list)
    annual_ordinary_income: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    annual_ltcg: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    prior_year_traditional_balance: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )
    annual_rmd_satisfied: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    current_spending: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    initial_portfolio_value: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([])
    )

    @classmethod
    def initialize(
        cls,
        n_paths: int,
        initial_balances: list[float],
        account_types: list[str],
        target_weights: NDArray[np.floating[Any]],
    ) -> SimulationState:
        """Create initial simulation state.

        Each account's balance is distributed across assets according to
        ``target_weights``.

        Args:
            n_paths: Number of simulation paths.
            initial_balances: Starting balance for each account.
            account_types: Account type labels.
            target_weights: (n_accounts, n_assets) target asset allocation weights summing to 1
                per account.
        """
        n_accounts = len(initial_balances)
        n_assets = target_weights.shape[1]
        # positions[p, a, k] = balance_a * weight_a,k
        bal = np.array(initial_balances)  # (n_accounts,)
        # (n_accounts, n_assets)
        account_positions = bal[:, np.newaxis] * target_weights
        # Tile across all paths → (n_paths, n_accounts, n_assets)
        positions = np.tile(account_positions, (n_paths, 1, 1))
        return cls(
            positions=positions,
            cumulative_inflation=np.ones(n_paths),
            is_depleted=np.zeros(n_paths, dtype=bool),
            step=0,
            n_paths=n_paths,
            n_accounts=n_accounts,
            n_assets=n_assets,
            account_types=account_types,
            annual_ordinary_income=np.zeros(n_paths),
            annual_ltcg=np.zeros(n_paths),
            prior_year_traditional_balance=np.zeros(n_paths),
            annual_rmd_satisfied=np.zeros(n_paths),
            current_spending=np.zeros(n_paths),
            initial_portfolio_value=np.zeros(n_paths),
        )

    @property
    def balances(self) -> NDArray[np.floating[Any]]:
        """Account balances, shape (n_paths, n_accounts).

        Derived from positions by summing over assets.
        """
        result: NDArray[np.floating[Any]] = self.positions.sum(axis=2)
        return result

    @property
    def total_wealth(self) -> NDArray[np.floating[Any]]:
        """Total wealth across all accounts, shape (n_paths,)."""
        result: NDArray[np.floating[Any]] = self.positions.sum(axis=(1, 2))
        return result
