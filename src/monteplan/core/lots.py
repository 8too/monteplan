"""Tax lots and realized capital gains of taxable accounts.

The engine keeps each account's value in ``SimulationState.positions``; the ledger only
tracks how that value splits into lots. A lot is a number of units of its account, and
the account's unit price is its balance divided by its units. Returns, fees and
rebalancing within an account change the unit price but not the units, so the ledger
only needs to see money moving into and out of taxable accounts: a purchase adds units
to a new lot, and a sale removes units from the lots in ``lot_selection`` order and
realizes the difference between their value and their cost basis as a capital gain.

The engine calls ``snapshot()`` before and ``record_flows()`` after each step that moves
money into or out of accounts. Each December, ``close_year()`` nets the year's gains and
losses and adds them to the state's ordinary income and long-term capital gains.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from monteplan.config.schema import AccountConfig, LotConfig
from monteplan.core.state import SimulationState

# Gains on lots held longer than this are long-term.
LONG_TERM_MONTHS = 12
# Net capital loss deductible from ordinary income per year; the rest carries forward.
CAPITAL_LOSS_LIMIT = 3000.0
# Balance changes below this fraction of the balance are rounding, not cash flows.
_FLOW_TOLERANCE = 1e-9


class LotLedger:
    """Tax lots of all taxable accounts, for all paths.

    Attributes:
        taxable: Indices of the taxable accounts.
        lot_account: (n_lots,) account index of each lot.
        acquired_step: (n_lots,) step of each lot's most recent purchase (negative for lots
            bought before the simulation start).
        units: (n_paths, n_lots) units of each lot. Units start at a price of 1.0.
        basis: (n_paths, n_lots) remaining nominal cost basis of each lot in dollars.
        short_term_gains: (n_paths,) net short-term gains realized so far this year.
        long_term_gains: (n_paths,) net long-term gains realized so far this year.
        loss_carryforward: (n_paths,) net capital losses carried into this year.
    """

    def __init__(
        self,
        n_paths: int,
        accounts: list[AccountConfig],
        lot_selection: Literal["fifo", "lifo", "hifo"],
    ) -> None:
        self.lot_selection = lot_selection
        self.taxable = [i for i, a in enumerate(accounts) if a.account_type == "taxable"]
        lot_account: list[int] = []
        acquired_step: list[int] = []
        units: list[float] = []
        basis: list[float] = []
        for i in self.taxable:
            lots = accounts[i].lots
            if lots is None:
                lots = [LotConfig(amount=accounts[i].balance, months_held=LONG_TERM_MONTHS + 1)]
            for lot in lots:
                lot_account.append(i)
                acquired_step.append(-lot.months_held)
                units.append(lot.amount)
                basis.append(lot.amount if lot.cost_basis is None else lot.cost_basis)
        self.lot_account = np.array(lot_account, dtype=int)
        self.acquired_step = np.array(acquired_step, dtype=int)
        self.units: NDArray[np.floating[Any]] = np.tile(np.array(units, dtype=float), (n_paths, 1))
        self.basis: NDArray[np.floating[Any]] = np.tile(np.array(basis, dtype=float), (n_paths, 1))
        self.short_term_gains = np.zeros(n_paths)
        self.long_term_gains = np.zeros(n_paths)
        self.loss_carryforward = np.zeros(n_paths)
        # Lot bought into by each (account, simulation year), so that each account gets at
        # most one new lot per year.
        self._new_lots: dict[tuple[int, int], int] = {}

    def snapshot(self, state: SimulationState) -> NDArray[np.floating[Any]] | None:
        """Taxable account balances before a step that moves money, for ``record_flows()``."""
        if not self.taxable:
            return None
        result: NDArray[np.floating[Any]] = state.balances[:, self.taxable]
        return result

    def record_flows(
        self,
        state: SimulationState,
        balances_before: NDArray[np.floating[Any]] | None,
        step: int,
    ) -> None:
        """Buys or sells lots for each change of a taxable balance since ``snapshot()``."""
        if balances_before is None:
            return
        balances_after = state.balances[:, self.taxable]
        for k, acct in enumerate(self.taxable):
            before = balances_before[:, k]
            delta = balances_after[:, k] - before
            tolerance = _FLOW_TOLERANCE * np.maximum(before, 1.0)
            if not (np.abs(delta) > tolerance).any():
                continue
            cols = self._columns(acct)
            # Lots of an account whose balance dropped to zero are worthless, and would
            # otherwise come back to life at the fresh unit price below.
            worthless = before <= 0.0
            if worthless.any():
                self.units[np.ix_(worthless, cols)] = 0.0
            total_units = self.units[:, cols].sum(axis=1)
            price = np.where(
                total_units > 0, before / np.where(total_units > 0, total_units, 1.0), 1.0
            )
            sold = np.where(delta < -tolerance, -delta, 0.0)
            if sold.any():
                self._sell(cols, sold, price, step)
            bought = np.where(delta > tolerance, delta, 0.0)
            if bought.any():
                self._buy(acct, bought, price, step)

    def close_year(self, state: SimulationState) -> None:
        """Adds the year's net realized gains to the state's annual income accumulators.

        Short- and long-term results net against each other. A net loss reduces ordinary
        income by up to ``CAPITAL_LOSS_LIMIT`` and carries the rest forward, where it is
        treated as a short-term loss.
        """
        short_term = self.short_term_gains - self.loss_carryforward
        long_term = self.long_term_gains
        total = short_term + long_term
        # The net gain is long-term up to the net long-term gain.
        net_long_term = np.where(short_term < 0, total, np.where(long_term < 0, 0.0, long_term))
        net_long_term = np.where(total > 0, net_long_term, 0.0)
        net_short_term = np.where(total > 0, total - net_long_term, 0.0)
        loss = np.maximum(-total, 0.0)
        deduction = np.minimum(loss, CAPITAL_LOSS_LIMIT)
        state.annual_ordinary_income = np.maximum(
            state.annual_ordinary_income + net_short_term - deduction, 0.0
        )
        state.annual_ltcg = state.annual_ltcg + net_long_term
        self.loss_carryforward = loss - deduction
        self.short_term_gains = np.zeros_like(self.short_term_gains)
        self.long_term_gains = np.zeros_like(self.long_term_gains)

    def lot_values(self, state: SimulationState) -> NDArray[np.floating[Any]]:
        """(n_paths, n_lots) current market value of each lot."""
        values = np.zeros_like(self.units)
        balances = state.balances
        for acct in self.taxable:
            cols = self._columns(acct)
            account_units = self.units[:, cols]
            total_units = account_units.sum(axis=1, keepdims=True)
            safe_units = np.where(total_units > 0, total_units, 1.0)
            values[:, cols] = account_units / safe_units * balances[:, acct, np.newaxis]
        return values

    def _columns(self, acct: int) -> NDArray[np.intp]:
        return np.flatnonzero(self.lot_account == acct)

    def _buy(
        self,
        acct: int,
        dollars: NDArray[np.floating[Any]],
        price: NDArray[np.floating[Any]],
        step: int,
    ) -> None:
        key = (acct, step // 12)
        col = self._new_lots.get(key)
        if col is None:
            col = self.units.shape[1]
            self._new_lots[key] = col
            self.lot_account = np.append(self.lot_account, acct)
            self.acquired_step = np.append(self.acquired_step, step)
            new_column = np.zeros((self.units.shape[0], 1))
            self.units = np.hstack([self.units, new_column])
            self.basis = np.hstack([self.basis, new_column])
        self.units[:, col] += dollars / price
        self.basis[:, col] += dollars
        # Later purchases into the same lot restart its holding period, so that it is never
        # overstated.
        self.acquired_step[col] = step

    def _sell(
        self,
        cols: NDArray[np.intp],
        dollars: NDArray[np.floating[Any]],
        price: NDArray[np.floating[Any]],
        step: int,
    ) -> None:
        units = self.units[:, cols]
        basis = self.basis[:, cols]
        acquired = self.acquired_step[cols]
        if self.lot_selection == "hifo":
            # All lots of an account have the same unit price, so the highest basis per
            # unit is the smallest gain per dollar sold.
            per_unit = np.where(units > 0, basis / np.where(units > 0, units, 1.0), -np.inf)
            order = np.argsort(-per_unit, axis=1, kind="stable")
        else:
            by_age = np.argsort(acquired, kind="stable")
            if self.lot_selection == "lifo":
                by_age = by_age[::-1]
            order = np.broadcast_to(by_age, units.shape)
        units_o = np.take_along_axis(units, order, axis=1)
        basis_o = np.take_along_axis(basis, order, axis=1)
        acquired_o = acquired[order]

        value = units_o * price[:, np.newaxis]
        value_before = np.cumsum(value, axis=1) - value
        sold = np.clip(dollars[:, np.newaxis] - value_before, 0.0, value)
        frac = np.divide(sold, value, out=np.zeros_like(value), where=value > 0)
        basis_sold = basis_o * frac
        gain = sold - basis_sold
        long_term = (step - acquired_o) > LONG_TERM_MONTHS
        self.long_term_gains += np.where(long_term, gain, 0.0).sum(axis=1)
        self.short_term_gains += np.where(long_term, 0.0, gain).sum(axis=1)

        np.put_along_axis(units, order, units_o * (1.0 - frac), axis=1)
        np.put_along_axis(basis, order, basis_o - basis_sold, axis=1)
        self.units[:, cols] = units
        self.basis[:, cols] = basis
