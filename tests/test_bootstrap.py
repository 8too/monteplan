"""Tests for historical block bootstrap return model."""

from __future__ import annotations

import numpy as np

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate
from monteplan.core.rng import make_rng
from monteplan.models.returns.bootstrap import HistoricalBootstrapReturns


class TestBootstrap:
    def test_output_shape(self) -> None:
        # 60 months of history, 2 assets
        historical = np.random.default_rng(42).standard_normal((60, 2)) * 0.04
        model = HistoricalBootstrapReturns(historical, block_size=12)
        result = model.sample(100, 120, make_rng(42))
        assert result.shape == (100, 120, 2)

    def test_values_from_historical(self) -> None:
        """All sampled values should exist in the historical data."""
        rng_data = np.random.default_rng(99)
        historical = rng_data.standard_normal((60, 1)) * 0.05
        model = HistoricalBootstrapReturns(historical, block_size=6)
        result = model.sample(10, 24, make_rng(42))
        # Each sampled return should be in the historical data
        for val in result.flat:
            assert val in historical

    def test_deterministic(self) -> None:
        """Same seed should produce same samples."""
        historical = np.random.default_rng(42).standard_normal((60, 2)) * 0.04
        model = HistoricalBootstrapReturns(historical, block_size=12)
        r1 = model.sample(10, 24, make_rng(42))
        r2 = model.sample(10, 24, make_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_block_size_preserved(self) -> None:
        """Contiguous blocks should appear in the output."""
        historical = np.arange(48).reshape(48, 1).astype(float)
        model = HistoricalBootstrapReturns(historical, block_size=6)
        result = model.sample(1, 12, make_rng(42))
        # Each block of 6 should be contiguous from historical
        for block_start in range(0, 12, 6):
            block = result[0, block_start : block_start + 6, 0]
            diffs = np.diff(block)
            assert np.all(diffs == 1.0)  # consecutive integers

    def test_engine_integration(self) -> None:
        """Bootstrap model should work in the full simulation."""
        rng_data = np.random.default_rng(42)
        historical = (rng_data.standard_normal((120, 1)) * 0.04 + 0.005).tolist()
        plan = PlanConfig(
            current_age=60,
            retirement_age=65,
            end_age=75,
            accounts=[AccountConfig(balance=500_000)],
            monthly_income=5_000,
            monthly_spending=3_000,
        )
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
            expected_annual_returns=[0.06],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
            return_model="bootstrap",
            historical_returns=historical,
            bootstrap_block_size=12,
        )
        result = simulate(plan, market, PolicyBundle(), SimulationConfig(n_paths=50, seed=42))
        assert 0.0 <= result.success_probability <= 1.0
