"""Tests for rebalancing policy."""

from __future__ import annotations

import numpy as np

from monteplan.core.state import SimulationState
from monteplan.policies.rebalancing import rebalance_to_targets


class TestRebalancing:
    def test_preserves_total_balance(self) -> None:
        """Rebalancing should preserve total account balance."""
        # 1 path, 1 account, 2 assets: drifted to 80/20 from 70/30
        positions = np.array([[[80_000.0, 20_000.0]]])
        state = SimulationState(
            positions=positions,
            cumulative_inflation=np.ones(1),
            is_depleted=np.zeros(1, dtype=bool),
            step=0,
            n_paths=1,
            n_accounts=1,
            n_assets=2,
            account_types=["taxable"],
        )
        target_weights = np.array([[0.7, 0.3]])
        balance_before = state.balances[0, 0]
        rebalance_to_targets(state, target_weights)
        np.testing.assert_allclose(state.balances[0, 0], balance_before)

    def test_rebalances_to_target_weights(self) -> None:
        """After rebalancing, positions should match target weights."""
        positions = np.array([[[80_000.0, 20_000.0]]])
        state = SimulationState(
            positions=positions,
            cumulative_inflation=np.ones(1),
            is_depleted=np.zeros(1, dtype=bool),
            step=0,
            n_paths=1,
            n_accounts=1,
            n_assets=2,
            account_types=["taxable"],
        )
        target_weights = np.array([[0.7, 0.3]])
        rebalance_to_targets(state, target_weights)
        # Total = 100k, so stocks=70k, bonds=30k
        np.testing.assert_allclose(state.positions[0, 0, 0], 70_000.0)
        np.testing.assert_allclose(state.positions[0, 0, 1], 30_000.0)

    def test_multiple_accounts(self) -> None:
        """Each account is rebalanced independently."""
        positions = np.array([[[90_000.0, 10_000.0], [40_000.0, 60_000.0]]])
        state = SimulationState(
            positions=positions,
            cumulative_inflation=np.ones(1),
            is_depleted=np.zeros(1, dtype=bool),
            step=0,
            n_paths=1,
            n_accounts=2,
            n_assets=2,
            account_types=["taxable", "traditional"],
        )
        target_weights = np.array([[0.6, 0.4], [0.6, 0.4]])
        rebalance_to_targets(state, target_weights)
        # Account 0: total=100k → 60k/40k
        np.testing.assert_allclose(state.positions[0, 0, :], [60_000.0, 40_000.0])
        # Account 1: total=100k → 60k/40k
        np.testing.assert_allclose(state.positions[0, 1, :], [60_000.0, 40_000.0])

    def test_vectorized_paths(self) -> None:
        """Rebalancing works across multiple paths."""
        positions = np.array(
            [
                [[80_000.0, 20_000.0]],
                [[50_000.0, 50_000.0]],
            ]
        )
        state = SimulationState(
            positions=positions,
            cumulative_inflation=np.ones(2),
            is_depleted=np.zeros(2, dtype=bool),
            step=0,
            n_paths=2,
            n_accounts=1,
            n_assets=2,
            account_types=["taxable"],
        )
        target_weights = np.array([[0.7, 0.3]])
        rebalance_to_targets(state, target_weights)
        # Path 0: total=100k → 70k/30k
        np.testing.assert_allclose(state.positions[0, 0, :], [70_000.0, 30_000.0])
        # Path 1: total=100k → 70k/30k
        np.testing.assert_allclose(state.positions[1, 0, :], [70_000.0, 30_000.0])
