"""Tests for simulation state with per-asset-per-account positions."""

from __future__ import annotations

import numpy as np

from monteplan.core.state import SimulationState


class TestSimulationState:
    def test_initialize_shape(self) -> None:
        """Positions should have shape (n_paths, n_accounts, n_assets)."""
        weights = np.array([[0.7, 0.3], [0.7, 0.3]])
        state = SimulationState.initialize(
            n_paths=100,
            initial_balances=[50_000.0, 100_000.0],
            account_types=["taxable", "traditional"],
            target_weights=weights,
        )
        assert state.positions.shape == (100, 2, 2)
        assert state.n_assets == 2

    def test_initialize_allocation(self) -> None:
        """Initial positions should match balance * target_weights."""
        weights = np.array([[0.6, 0.4]])
        state = SimulationState.initialize(
            n_paths=1,
            initial_balances=[100_000.0],
            account_types=["taxable"],
            target_weights=weights,
        )
        np.testing.assert_allclose(state.positions[0, 0, :], [60_000.0, 40_000.0])

    def test_balances_derived(self) -> None:
        """Balances should be sum of positions across assets."""
        weights = np.array([[0.7, 0.3], [0.7, 0.3]])
        state = SimulationState.initialize(
            n_paths=2,
            initial_balances=[100_000.0, 50_000.0],
            account_types=["taxable", "traditional"],
            target_weights=weights,
        )
        expected = np.array([[100_000.0, 50_000.0], [100_000.0, 50_000.0]])
        np.testing.assert_allclose(state.balances, expected)

    def test_total_wealth(self) -> None:
        """Total wealth should sum all positions."""
        weights = np.array([[0.7, 0.3], [0.7, 0.3]])
        state = SimulationState.initialize(
            n_paths=2,
            initial_balances=[100_000.0, 50_000.0],
            account_types=["taxable", "traditional"],
            target_weights=weights,
        )
        np.testing.assert_allclose(state.total_wealth, [150_000.0, 150_000.0])

    def test_positions_uniform_across_paths(self) -> None:
        """All paths should start with identical positions."""
        weights = np.array([[0.5, 0.5]])
        state = SimulationState.initialize(
            n_paths=50,
            initial_balances=[80_000.0],
            account_types=["taxable"],
            target_weights=weights,
        )
        for p in range(50):
            np.testing.assert_allclose(state.positions[p, 0, :], [40_000.0, 40_000.0])
