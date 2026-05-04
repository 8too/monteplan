"""Tests for floor-and-ceiling spending policy (G4)."""

import numpy as np

from monteplan.config.schema import FloorCeilingConfig
from monteplan.core.state import SimulationState
from monteplan.policies.spending.floor_ceiling import FloorCeilingSpending


def _make_state(n_paths, total_wealth, cumulative_inflation=1.0):
    """Create a minimal SimulationState for spending tests."""
    weights = np.array([[1.0]])
    state = SimulationState.initialize(
        n_paths=n_paths,
        initial_balances=[0.0],
        account_types=["taxable"],
        target_weights=weights,
    )
    # Set positions to desired total wealth
    state.positions[:, 0, 0] = total_wealth
    state.cumulative_inflation[:] = cumulative_inflation
    return state


def test_spending_between_floor_and_ceiling():
    """When rate*wealth is between floor and ceiling, use it as-is."""
    config = FloorCeilingConfig(withdrawal_rate=0.04, floor=2000, ceiling=10000)
    policy = FloorCeilingSpending(config)
    # 4%/12 * 1_500_000 = 5000, between 2000 and 10000
    state = _make_state(10, 1_500_000)
    spending = policy.compute(state)
    expected = 0.04 / 12 * 1_500_000
    np.testing.assert_allclose(spending, expected, rtol=1e-10)


def test_floor_binds():
    """When rate*wealth < floor, spending should equal the floor."""
    config = FloorCeilingConfig(withdrawal_rate=0.04, floor=3000, ceiling=10000)
    policy = FloorCeilingSpending(config)
    # 4%/12 * 100_000 = 333, well below 3000 floor
    state = _make_state(10, 100_000)
    spending = policy.compute(state)
    np.testing.assert_allclose(spending, 3000.0, rtol=1e-10)


def test_ceiling_binds():
    """When rate*wealth > ceiling, spending should equal the ceiling."""
    config = FloorCeilingConfig(withdrawal_rate=0.04, floor=2000, ceiling=5000)
    policy = FloorCeilingSpending(config)
    # 4%/12 * 5_000_000 = 16667, well above 5000 ceiling
    state = _make_state(10, 5_000_000)
    spending = policy.compute(state)
    np.testing.assert_allclose(spending, 5000.0, rtol=1e-10)


def test_floor_ceiling_inflation_adjusted():
    """Floor and ceiling should scale with cumulative inflation."""
    config = FloorCeilingConfig(withdrawal_rate=0.04, floor=3000, ceiling=10000)
    policy = FloorCeilingSpending(config)
    inflation = 1.5
    # 4%/12 * 100_000 = 333, below floor of 3000*1.5=4500
    state = _make_state(10, 100_000, cumulative_inflation=inflation)
    spending = policy.compute(state)
    np.testing.assert_allclose(spending, 3000.0 * inflation, rtol=1e-10)


def test_zero_wealth_returns_floor():
    """With zero wealth, spending should be the floor."""
    config = FloorCeilingConfig(withdrawal_rate=0.04, floor=2000, ceiling=10000)
    policy = FloorCeilingSpending(config)
    state = _make_state(5, 0.0)
    spending = policy.compute(state)
    np.testing.assert_allclose(spending, 2000.0, rtol=1e-10)
