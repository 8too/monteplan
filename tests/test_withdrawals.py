"""Tests for withdrawal ordering."""

from __future__ import annotations

import numpy as np

from monteplan.core.state import SimulationState
from monteplan.policies.withdrawals import pay_tax, withdraw


def _make_state(
    balances: list[list[float]],
    account_types: list[str],
) -> SimulationState:
    """Create a state with positions from balances (single-asset for simplicity)."""
    n_paths = len(balances)
    bal = np.array(balances, dtype=float)
    n_accounts = bal.shape[1]
    # Use single asset so positions[:, :, 0] == balances
    positions = bal[:, :, np.newaxis]  # (n_paths, n_accounts, 1)
    return SimulationState(
        positions=positions,
        cumulative_inflation=np.ones(n_paths),
        is_depleted=np.zeros(n_paths, dtype=bool),
        step=0,
        n_paths=n_paths,
        n_accounts=n_accounts,
        n_assets=1,
        account_types=account_types,
    )


class TestWithdrawals:
    def test_taxable_first(self) -> None:
        """Should withdraw from taxable before traditional."""
        state = _make_state(
            [[50_000, 100_000, 30_000]],
            ["taxable", "traditional", "roth"],
        )
        need = np.array([10_000.0])
        got = withdraw(state, need, ["taxable", "traditional", "roth"], tax_rate=0.22)
        np.testing.assert_allclose(got, [10_000.0])
        # Should come entirely from taxable
        np.testing.assert_allclose(state.balances[0, 0], 40_000.0)
        np.testing.assert_allclose(state.balances[0, 1], 100_000.0)
        np.testing.assert_allclose(state.balances[0, 2], 30_000.0)

    def test_traditional_grossed_up(self) -> None:
        """Traditional withdrawals should be grossed up for taxes."""
        state = _make_state(
            [[0, 100_000, 0]],
            ["taxable", "traditional", "roth"],
        )
        need = np.array([7_800.0])  # after-tax need
        tax_rate = 0.22
        got = withdraw(state, need, ["taxable", "traditional", "roth"], tax_rate=tax_rate)
        np.testing.assert_allclose(got, [7_800.0])
        # Gross withdrawal = 7800 / (1 - 0.22) = 10000
        np.testing.assert_allclose(state.balances[0, 1], 90_000.0)

    def test_cascade_through_accounts(self) -> None:
        """When taxable is depleted, should cascade to traditional."""
        state = _make_state(
            [[5_000, 100_000, 50_000]],
            ["taxable", "traditional", "roth"],
        )
        need = np.array([8_000.0])
        got = withdraw(state, need, ["taxable", "traditional", "roth"], tax_rate=0.22)
        np.testing.assert_allclose(got, [8_000.0])
        # Taxable fully depleted
        np.testing.assert_allclose(state.balances[0, 0], 0.0)
        # Remaining 3000 after-tax from traditional = 3000 / 0.78 gross
        expected_trad = 100_000 - 3_000 / 0.78
        np.testing.assert_allclose(state.balances[0, 1], expected_trad, atol=0.01)

    def test_no_negative_balances(self) -> None:
        """Should never produce negative balances."""
        state = _make_state(
            [[1_000, 2_000, 500]],
            ["taxable", "traditional", "roth"],
        )
        need = np.array([100_000.0])  # way more than available
        got = withdraw(state, need, ["taxable", "traditional", "roth"], tax_rate=0.22)
        assert np.all(state.balances >= -1e-10)
        assert got[0] < 100_000.0  # couldn't fulfill full need

    def test_vectorized(self) -> None:
        """Should work correctly across multiple paths."""
        state = _make_state(
            [[50_000, 100_000], [10_000, 20_000]],
            ["taxable", "traditional"],
        )
        need = np.array([5_000.0, 5_000.0])
        got = withdraw(state, need, ["taxable", "traditional"], tax_rate=0.20)
        np.testing.assert_allclose(got, [5_000.0, 5_000.0])

    def test_pro_rata_selling(self) -> None:
        """Withdrawal should reduce positions pro-rata across assets."""
        n_paths = 1
        # 2 accounts, 2 assets: taxable has 60k stocks + 40k bonds
        positions = np.array([[[60_000.0, 40_000.0], [50_000.0, 50_000.0]]])
        state = SimulationState(
            positions=positions,
            cumulative_inflation=np.ones(n_paths),
            is_depleted=np.zeros(n_paths, dtype=bool),
            step=0,
            n_paths=n_paths,
            n_accounts=2,
            n_assets=2,
            account_types=["taxable", "traditional"],
        )
        # Withdraw 50k from taxable (50% of 100k account)
        need = np.array([50_000.0])
        withdraw(state, need, ["taxable", "traditional"], tax_rate=0.22)
        # Taxable should have 50% of original positions
        np.testing.assert_allclose(state.positions[0, 0, 0], 30_000.0)
        np.testing.assert_allclose(state.positions[0, 0, 1], 20_000.0)
        # Traditional untouched
        np.testing.assert_allclose(state.positions[0, 1, 0], 50_000.0)
        np.testing.assert_allclose(state.positions[0, 1, 1], 50_000.0)


class TestPayTax:
    """Year-end tax comes from taxable, then traditional, and Roth only once both are empty."""

    def _pay(self, tax: float) -> tuple[np.ndarray, np.ndarray]:
        state = _make_state([[50_000, 100_000, 30_000]], ["taxable", "traditional", "roth"])
        from_traditional = pay_tax(state, np.array([tax], dtype=float))
        return state.balances[0], from_traditional

    def test_taxable_first(self) -> None:
        balances, from_traditional = self._pay(20_000)
        np.testing.assert_allclose(balances, [30_000, 100_000, 30_000])
        np.testing.assert_allclose(from_traditional, [0.0])

    def test_then_traditional(self) -> None:
        balances, from_traditional = self._pay(70_000)
        np.testing.assert_allclose(balances, [0, 80_000, 30_000])
        np.testing.assert_allclose(from_traditional, [20_000])

    def test_roth_last(self) -> None:
        balances, from_traditional = self._pay(160_000)
        np.testing.assert_allclose(balances, [0, 0, 20_000])
        np.testing.assert_allclose(from_traditional, [100_000])

    def test_shortfall_takes_everything(self) -> None:
        balances, _ = self._pay(200_000)
        np.testing.assert_allclose(balances, [0, 0, 0])
