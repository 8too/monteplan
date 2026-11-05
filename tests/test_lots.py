"""Tests for tax lots and realized capital gains."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from monteplan.config.defaults import default_market
from monteplan.config.schema import (
    AccountConfig,
    LotConfig,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate
from monteplan.core.lots import LotLedger
from monteplan.core.state import SimulationState


def _setup(
    lots: list[LotConfig], n_paths: int = 1, lot_selection: str = "fifo"
) -> tuple[SimulationState, LotLedger]:
    """A single-asset state with one taxable account holding the given lots."""
    balance = sum(lot.amount for lot in lots)
    accounts = [AccountConfig(account_type="taxable", balance=balance, lots=lots)]
    state = SimulationState.initialize(n_paths, [balance], ["taxable"], np.ones((1, 1)))
    return state, LotLedger(n_paths, accounts, lot_selection)  # type: ignore[arg-type]


def _flow(state: SimulationState, ledger: LotLedger, amount: float, step: int) -> None:
    """Adds (or, if negative, withdraws) money and records it in the ledger."""
    before = ledger.snapshot(state)
    state.positions[:, 0, 0] += amount
    ledger.record_flows(state, before, step)


class TestLotSales:
    # Oldest first: A (gain 40, long-term), B (gain 5, long-term), C (gain 30, short-term).
    LOTS = [
        LotConfig(amount=100, cost_basis=60, months_held=36),
        LotConfig(amount=100, cost_basis=95, months_held=24),
        LotConfig(amount=100, cost_basis=70, months_held=3),
    ]

    @pytest.mark.parametrize(
        ("lot_selection", "sold_lot", "short_term", "long_term"),
        [("fifo", 0, 0.0, 40.0), ("lifo", 2, 30.0, 0.0), ("hifo", 1, 0.0, 5.0)],
    )
    def test_sale_order(
        self, lot_selection: str, sold_lot: int, short_term: float, long_term: float
    ) -> None:
        state, ledger = _setup(self.LOTS, lot_selection=lot_selection)
        _flow(state, ledger, -100.0, step=0)
        np.testing.assert_allclose(ledger.short_term_gains, [short_term])
        np.testing.assert_allclose(ledger.long_term_gains, [long_term])
        expected_units = [100.0, 100.0, 100.0]
        expected_units[sold_lot] = 0.0
        np.testing.assert_allclose(ledger.units[0], expected_units)

    def test_sale_spans_lots(self) -> None:
        """Sells whole lots first, then part of the next one with its share of the basis."""
        state, ledger = _setup(self.LOTS)
        _flow(state, ledger, -150.0, step=0)
        np.testing.assert_allclose(ledger.long_term_gains, [40.0 + 2.5])
        np.testing.assert_allclose(ledger.units[0], [0.0, 50.0, 100.0])
        np.testing.assert_allclose(ledger.basis[0], [0.0, 47.5, 70.0])

    def test_hifo_order_is_per_path(self) -> None:
        state, ledger = _setup(
            [
                LotConfig(amount=100, cost_basis=50, months_held=24),
                LotConfig(amount=100, cost_basis=80, months_held=24),
            ],
            n_paths=2,
            lot_selection="hifo",
        )
        ledger.basis[1] = [90.0, 80.0]
        _flow(state, ledger, -100.0, step=0)
        np.testing.assert_allclose(ledger.long_term_gains, [20.0, 10.0])
        np.testing.assert_allclose(ledger.units, [[100.0, 0.0], [0.0, 100.0]])

    def test_gains_use_current_price(self) -> None:
        state, ledger = _setup([LotConfig(amount=100, cost_basis=100, months_held=24)])
        state.positions *= 2.0  # Returns change the unit price, not the units
        _flow(state, ledger, -50.0, step=0)
        np.testing.assert_allclose(ledger.long_term_gains, [25.0])
        np.testing.assert_allclose(ledger.units[0], [75.0])
        np.testing.assert_allclose(ledger.basis[0], [75.0])
        np.testing.assert_allclose(ledger.lot_values(state)[0], [150.0])

    @pytest.mark.parametrize(("step", "long_term"), [(6, False), (7, True)])
    def test_long_term_after_twelve_months(self, step: int, long_term: bool) -> None:
        state, ledger = _setup([LotConfig(amount=100, cost_basis=80, months_held=6)])
        _flow(state, ledger, -100.0, step=step)
        np.testing.assert_allclose(ledger.long_term_gains, [20.0 if long_term else 0.0])
        np.testing.assert_allclose(ledger.short_term_gains, [0.0 if long_term else 20.0])

    def test_unchanged_balance_records_nothing(self) -> None:
        state, ledger = _setup(self.LOTS)
        _flow(state, ledger, 0.0, step=0)
        assert ledger.units.shape == (1, 3)
        np.testing.assert_allclose(ledger.short_term_gains + ledger.long_term_gains, [0.0])


class TestLotPurchases:
    def test_purchases_create_one_lot_per_year(self) -> None:
        state, ledger = _setup([LotConfig(amount=100, cost_basis=100, months_held=24)])
        state.positions *= 2.0
        _flow(state, ledger, 100.0, step=30)
        _flow(state, ledger, 100.0, step=31)
        _flow(state, ledger, 100.0, step=36)
        np.testing.assert_allclose(ledger.units[0], [100.0, 100.0, 50.0])
        np.testing.assert_allclose(ledger.basis[0], [100.0, 200.0, 100.0])
        # The second purchase restarts the holding period of the year's lot.
        np.testing.assert_array_equal(ledger.acquired_step, [-24, 31, 36])
        np.testing.assert_allclose(ledger.lot_values(state)[0], [200.0, 200.0, 100.0])

    def test_worthless_lots_stay_worthless(self) -> None:
        state, ledger = _setup([LotConfig(amount=100, cost_basis=100, months_held=24)])
        state.positions *= 0.0
        _flow(state, ledger, 50.0, step=1)
        np.testing.assert_allclose(ledger.lot_values(state)[0], [0.0, 50.0])

    def test_account_without_lots_is_one_long_term_lot_at_cost(self) -> None:
        accounts = [AccountConfig(account_type="taxable", balance=100.0)]
        state = SimulationState.initialize(1, [100.0], ["taxable"], np.ones((1, 1)))
        ledger = LotLedger(1, accounts, "fifo")
        state.positions *= 1.5
        _flow(state, ledger, -150.0, step=0)
        np.testing.assert_allclose(ledger.long_term_gains, [50.0])


class TestCloseYear:
    def test_netting_deduction_and_carryforward(self) -> None:
        state, ledger = _setup([LotConfig(amount=100, months_held=24)], n_paths=4)
        state.annual_ordinary_income[:] = 10_000.0
        ledger.short_term_gains = np.array([-1_000.0, 5_000.0, -10_000.0, 1_000.0])
        ledger.long_term_gains = np.array([3_000.0, -2_000.0, 2_000.0, 4_000.0])
        ledger.close_year(state)
        np.testing.assert_allclose(state.annual_ordinary_income, [10_000, 13_000, 7_000, 11_000])
        np.testing.assert_allclose(state.annual_ltcg, [2_000, 0, 0, 4_000])
        np.testing.assert_allclose(ledger.loss_carryforward, [0, 0, 5_000, 0])
        np.testing.assert_allclose(ledger.short_term_gains + ledger.long_term_gains, 0.0)

        # The carried-forward loss offsets next year's gains.
        state.annual_ordinary_income[:] = 0.0
        state.annual_ltcg[:] = 0.0
        ledger.long_term_gains = np.array([0.0, 0.0, 6_000.0, 0.0])
        ledger.close_year(state)
        np.testing.assert_allclose(state.annual_ltcg, [0, 0, 1_000, 0])
        np.testing.assert_allclose(ledger.loss_carryforward, 0.0)


class TestLotConfig:
    def test_lots_must_sum_to_balance(self) -> None:
        with pytest.raises(ValidationError, match="sum to"):
            AccountConfig(balance=100.0, lots=[LotConfig(amount=90.0, months_held=1)])

    def test_only_taxable_accounts_have_lots(self) -> None:
        with pytest.raises(ValidationError, match="Only taxable"):
            AccountConfig(
                account_type="roth", balance=100.0, lots=[LotConfig(amount=100.0, months_held=1)]
            )


class TestEngine:
    @staticmethod
    def _success(tax_model: str, cost_basis: float | None) -> float:
        lots = (
            None
            if cost_basis is None
            else [LotConfig(amount=1e6, cost_basis=cost_basis, months_held=24)]
        )
        plan = PlanConfig(
            current_age=60,
            retirement_age=60,
            end_age=90,
            monthly_spending=4_000,
            accounts=[AccountConfig(account_type="taxable", balance=1e6, lots=lots)],
        )
        result = simulate(
            plan,
            default_market(),
            PolicyBundle(tax_model=tax_model),  # type: ignore[arg-type]
            SimulationConfig(n_paths=500, seed=7),
        )
        return result.success_probability

    def test_us_federal_taxes_realized_gains(self) -> None:
        """Selling low-basis lots costs capital gains tax (upstream issue #2)."""
        assert self._success("us_federal", 0.0) < self._success("us_federal", 1e6)

    def test_default_basis_is_the_balance(self) -> None:
        assert self._success("us_federal", None) == self._success("us_federal", 1e6)
