"""Rebalancing policies: calendar and threshold-based."""

from __future__ import annotations

import numpy as np

from monteplan.core.state import SimulationState


def rebalance_to_targets(
    state: SimulationState,
    target_weights: np.ndarray,
) -> None:
    """Rebalance all accounts to target allocation weights.

    For each account, redistributes positions across assets so that
    the dollar allocation matches ``target_weights`` while preserving
    the total account balance. Vectorized across all paths.

    Args:
        state: Current simulation state (positions will be mutated).
        target_weights: (n_accounts, n_assets) target allocation weights summing to 1 per account.
    """
    # balances shape: (n_paths, n_accounts)
    balances = state.balances
    # target positions: balances[:, :, np.newaxis] * target_weights
    # → (n_paths, n_accounts, n_assets)
    state.positions = balances[:, :, np.newaxis] * target_weights[np.newaxis, :, :]


def rebalance_if_drifted(
    state: SimulationState,
    target_weights: np.ndarray,
    threshold: float,
) -> None:
    """Rebalance only paths where any account drifted beyond threshold.

    For each path and account, compute the current weights. If any
    asset's weight deviates from its target by more than ``threshold``,
    rebalance that entire path to targets. Paths within tolerance
    are left untouched.

    Args:
        state: Current simulation state (positions will be mutated).
        target_weights: (n_accounts, n_assets) target allocation weights summing to 1 per account.
        threshold: Maximum allowed absolute drift (e.g. 0.05 for 5%).
    """
    # balances shape: (n_paths, n_accounts)
    balances = state.balances
    safe_balances = np.where(balances > 0, balances, 1.0)
    # current_weights: (n_paths, n_accounts, n_assets)
    current_weights = state.positions / safe_balances[:, :, np.newaxis]

    # Check which paths have drifted beyond threshold
    # drift: (n_paths, n_accounts, n_assets)
    drift = np.abs(current_weights - target_weights[np.newaxis, :, :])
    # needs_rebalance: (n_paths,) if ANY account drifted
    needs_rebalance = np.any(drift > threshold, axis=(1, 2))

    if not needs_rebalance.any():
        return

    # Rebalance only drifted paths
    new_positions = balances[:, :, np.newaxis] * target_weights[np.newaxis, :, :]
    state.positions = np.where(
        needs_rebalance[:, np.newaxis, np.newaxis],
        new_positions,
        state.positions,
    )
