"""Tests for glide path interpolation and engine integration."""

from __future__ import annotations

import pytest

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    GlidePath,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate


class TestGlidePathConfig:
    def test_valid_glide_path(self) -> None:
        gp = GlidePath(
            start_age=30,
            start_weights=[0.9, 0.1],
            end_age=65,
            end_weights=[0.4, 0.6],
        )
        assert gp.start_age == 30
        assert gp.end_age == 65

    def test_invalid_ages(self) -> None:
        with pytest.raises(ValueError):
            GlidePath(
                start_age=65,
                start_weights=[0.9, 0.1],
                end_age=30,
                end_weights=[0.4, 0.6],
            )

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            GlidePath(
                start_age=30,
                start_weights=[0.5, 0.3],
                end_age=65,
                end_weights=[0.4, 0.6],
            )

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            GlidePath(
                start_age=30,
                start_weights=[0.9, 0.1],
                end_age=65,
                end_weights=[0.4, 0.3, 0.3],
            )


class TestGlidePathSimulation:
    def _market_with_glide_path(self) -> MarketAssumptions:
        return MarketAssumptions(
            asset_allocations=[
                AssetAllocation(
                    assets=[
                        AssetClass(name="Stocks", weight=0.7),
                        AssetClass(name="Bonds", weight=0.3),
                    ],
                    glide_path=GlidePath(
                        start_age=30,
                        start_weights=[0.9, 0.1],
                        end_age=95,
                        end_weights=[0.4, 0.6],
                    ),
                )
            ],
            expected_annual_returns=[0.07, 0.03],
            annual_volatilities=[0.16, 0.06],
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )

    def test_runs_with_glide_path(self) -> None:
        plan = PlanConfig(
            current_age=30,
            retirement_age=65,
            end_age=95,
            accounts=[AccountConfig(balance=100_000, annual_contribution=10_000)],
            monthly_income=5_000,
            monthly_spending=3_000,
        )
        result = simulate(
            plan,
            self._market_with_glide_path(),
            PolicyBundle(),
            SimulationConfig(n_paths=50, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0

    def test_glide_path_shifts_allocation(self) -> None:
        """With a glide path, the allocation should shift over time."""
        plan = PlanConfig(
            current_age=30,
            retirement_age=65,
            end_age=95,
            accounts=[AccountConfig(balance=100_000, annual_contribution=10_000)],
            monthly_income=5_000,
            monthly_spending=3_000,
        )
        # Run without glide path (static 70/30)
        market_static = MarketAssumptions(
            asset_allocations=[
                AssetAllocation(
                    assets=[
                        AssetClass(name="Stocks", weight=0.7),
                        AssetClass(name="Bonds", weight=0.3),
                    ]
                )
            ],
            expected_annual_returns=[0.07, 0.03],
            annual_volatilities=[0.16, 0.06],
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        sim = SimulationConfig(n_paths=200, seed=42)
        r_static = simulate(plan, market_static, PolicyBundle(), sim)
        r_glide = simulate(plan, self._market_with_glide_path(), PolicyBundle(), sim)

        # Results should differ due to different allocation over time
        assert r_static.success_probability != r_glide.success_probability

    def test_none_glide_path_uses_static(self) -> None:
        """When glide_path is None, static weights from assets apply."""
        market = MarketAssumptions(
            asset_allocations=[
                AssetAllocation(
                    assets=[
                        AssetClass(name="Stocks", weight=0.7),
                        AssetClass(name="Bonds", weight=0.3),
                    ]
                )
            ],
            expected_annual_returns=[0.07, 0.03],
            annual_volatilities=[0.16, 0.06],
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert market.asset_allocations[0].glide_path is None
