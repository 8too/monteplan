"""Contribution policy: fixed monthly contributions during accumulation."""

from __future__ import annotations

import numpy as np

from monteplan.core.state import SimulationState


def apply_contributions(
    state: SimulationState,
    monthly_contributions: list[float],
    target_weights: np.ndarray,
    growth_factor: float = 1.0,
) -> None:
    """Add monthly contributions to each account, allocated by target weights.

    Args:
        state: Current simulation state (positions will be mutated).
        monthly_contributions: Base monthly contribution for each account.
        target_weights: (n_accounts, n_assets) target asset allocation weights.
        growth_factor: Cumulative income growth factor (1.0 = no growth).
    """
    for i, contrib in enumerate(monthly_contributions):
        if contrib > 0:
            state.positions[:, i, :] += contrib * growth_factor * target_weights[i, :]


def compute_monthly_contributions(annual_contributions: list[float]) -> list[float]:
    """Convert annual contributions to monthly amounts."""
    return [c / 12.0 for c in annual_contributions]
