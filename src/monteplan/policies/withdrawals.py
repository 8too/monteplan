"""Withdrawal ordering policy: taxable -> traditional -> roth."""

from __future__ import annotations

import numpy as np

from monteplan.core.state import SimulationState


def withdraw(
    state: SimulationState,
    amount_needed: np.ndarray,
    withdrawal_order: list[str],
    tax_rate: float,
) -> np.ndarray:
    """Withdraw from accounts in priority order.

    For traditional accounts, gross up the withdrawal to cover taxes:
    to get $X after tax, withdraw $X / (1 - tax_rate).

    Positions are reduced pro-rata across assets within each account.

    Args:
        state: Current simulation state (positions will be mutated).
        amount_needed: (n_paths,) after-tax spending needed.
        withdrawal_order: Priority list of account types.
        tax_rate: Flat effective tax rate on traditional withdrawals.

    Returns:
        (n_paths,) actual after-tax amount withdrawn (may be less than needed
        if accounts are depleted).
    """
    remaining = amount_needed.copy()
    total_withdrawn_after_tax = np.zeros(state.n_paths)

    for acct_type in withdrawal_order:
        indices = [i for i, t in enumerate(state.account_types) if t == acct_type]
        if not indices:
            continue

        for idx in indices:
            # Account balance = sum of positions across assets
            available = state.positions[:, idx, :].sum(axis=1)  # (n_paths,)
            done = remaining <= 0

            if acct_type == "traditional":
                gross_factor = 1.0 / (1.0 - tax_rate)
                gross_needed = remaining * gross_factor
                gross_withdraw = np.minimum(gross_needed, available)
                after_tax = gross_withdraw * (1.0 - tax_rate)
            else:
                gross_withdraw = np.minimum(remaining, available)
                after_tax = gross_withdraw

            # Don't withdraw from paths that are done
            gross_withdraw = np.where(done, 0.0, gross_withdraw)
            after_tax = np.where(done, 0.0, after_tax)

            # Pro-rata reduce positions across assets
            # fraction = gross_withdraw / available (avoid divide-by-zero)
            safe_available = np.where(available > 0, available, 1.0)
            frac = gross_withdraw / safe_available  # fraction of account to sell
            frac = np.minimum(frac, 1.0)  # cap at 100%
            # positions[:, idx, :] *= (1 - frac[:, np.newaxis])
            state.positions[:, idx, :] *= 1.0 - frac[:, np.newaxis]

            remaining -= after_tax
            total_withdrawn_after_tax += after_tax

    return total_withdrawn_after_tax


def pay_tax(state: SimulationState, tax: np.ndarray) -> np.ndarray:
    """Pay tax from taxable accounts first, then traditional, and Roth only once both are empty.

    Positions are reduced pro-rata across assets within each account. A path that can't pay
    all of its tax ends up with zero wealth.

    Args:
        state: Current simulation state (positions will be mutated).
        tax: (n_paths,) tax due.

    Returns:
        (n_paths,) amount paid from traditional accounts, which is itself ordinary income.
    """
    trad_indices = [i for i, t in enumerate(state.account_types) if t == "traditional"]
    trad_before = state.balances[:, trad_indices].sum(axis=1)
    withdraw(state, tax, ["taxable", "traditional", "roth"], tax_rate=0.0)
    trad_after = state.balances[:, trad_indices].sum(axis=1)
    paid_from_traditional: np.ndarray = np.maximum(trad_before - trad_after, 0.0)
    return paid_from_traditional
