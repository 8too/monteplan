"""Tests for regime-switching return and inflation models."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.config.defaults import default_plan, default_policies
from monteplan.config.schema import (
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    RegimeConfig,
    RegimeSwitchingConfig,
    SimulationConfig,
)
from monteplan.core.engine import simulate
from monteplan.core.rng import make_rng
from monteplan.models.inflation import RegimeSwitchingInflationModel
from monteplan.models.returns.regime_switching import RegimeSwitchingReturns


def _make_rs_config(n_regimes: int = 2) -> RegimeSwitchingConfig:
    """Helper: create a valid regime-switching config."""
    if n_regimes == 2:
        return RegimeSwitchingConfig(
            regimes=[
                RegimeConfig(
                    name="bull",
                    expected_annual_returns=[0.12, 0.05],
                    annual_volatilities=[0.12, 0.04],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.025,
                    inflation_vol=0.008,
                ),
                RegimeConfig(
                    name="bear",
                    expected_annual_returns=[-0.05, 0.01],
                    annual_volatilities=[0.25, 0.08],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.045,
                    inflation_vol=0.02,
                ),
            ],
            transition_matrix=[[0.97, 0.03], [0.05, 0.95]],
            initial_regime=0,
        )
    else:
        return RegimeSwitchingConfig(
            regimes=[
                RegimeConfig(
                    name="bull",
                    expected_annual_returns=[0.12, 0.05],
                    annual_volatilities=[0.12, 0.04],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.025,
                    inflation_vol=0.008,
                ),
                RegimeConfig(
                    name="normal",
                    expected_annual_returns=[0.07, 0.03],
                    annual_volatilities=[0.16, 0.06],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.03,
                    inflation_vol=0.01,
                ),
                RegimeConfig(
                    name="bear",
                    expected_annual_returns=[-0.05, 0.01],
                    annual_volatilities=[0.25, 0.08],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.045,
                    inflation_vol=0.02,
                ),
            ],
            transition_matrix=[
                [0.95, 0.04, 0.01],
                [0.03, 0.94, 0.03],
                [0.02, 0.05, 0.93],
            ],
            initial_regime=1,
        )


class TestRegimeSwitchingConfig:
    def test_valid_config(self) -> None:
        cfg = _make_rs_config(2)
        assert len(cfg.regimes) == 2

    def test_transition_row_sum(self) -> None:
        with pytest.raises(ValueError, match="sums to"):
            RegimeSwitchingConfig(
                regimes=[
                    RegimeConfig(
                        name="a",
                        expected_annual_returns=[0.07],
                        annual_volatilities=[0.16],
                        correlation_matrix=[[1.0]],
                        inflation_mean=0.03,
                        inflation_vol=0.01,
                    ),
                    RegimeConfig(
                        name="b",
                        expected_annual_returns=[0.03],
                        annual_volatilities=[0.06],
                        correlation_matrix=[[1.0]],
                        inflation_mean=0.03,
                        inflation_vol=0.01,
                    ),
                ],
                transition_matrix=[[0.5, 0.3], [0.4, 0.6]],
            )

    def test_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="transition_matrix"):
            RegimeSwitchingConfig(
                regimes=[
                    RegimeConfig(
                        name="a",
                        expected_annual_returns=[0.07],
                        annual_volatilities=[0.16],
                        correlation_matrix=[[1.0]],
                        inflation_mean=0.03,
                        inflation_vol=0.01,
                    ),
                    RegimeConfig(
                        name="b",
                        expected_annual_returns=[0.03],
                        annual_volatilities=[0.06],
                        correlation_matrix=[[1.0]],
                        inflation_mean=0.03,
                        inflation_vol=0.01,
                    ),
                ],
                transition_matrix=[[1.0]],
            )

    def test_asset_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="assets"):
            RegimeSwitchingConfig(
                regimes=[
                    RegimeConfig(
                        name="a",
                        expected_annual_returns=[0.07, 0.03],
                        annual_volatilities=[0.16, 0.06],
                        correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                        inflation_mean=0.03,
                        inflation_vol=0.01,
                    ),
                    RegimeConfig(
                        name="b",
                        expected_annual_returns=[0.03],  # mismatch
                        annual_volatilities=[0.06],
                        correlation_matrix=[[1.0]],
                        inflation_mean=0.03,
                        inflation_vol=0.01,
                    ),
                ],
                transition_matrix=[[0.9, 0.1], [0.1, 0.9]],
            )


class TestRegimeSwitchingReturns:
    def test_output_shape(self) -> None:
        cfg = _make_rs_config(2)
        model = RegimeSwitchingReturns(cfg)
        rng = make_rng(42)
        result = model.sample(100, 60, rng)
        assert result.shape == (100, 60, 2)

    def test_determinism(self) -> None:
        cfg = _make_rs_config(3)
        model1 = RegimeSwitchingReturns(cfg)
        model2 = RegimeSwitchingReturns(cfg)
        r1 = model1.sample(200, 120, make_rng(99))
        r2 = model2.sample(200, 120, make_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_regime_indices_stored(self) -> None:
        cfg = _make_rs_config(2)
        model = RegimeSwitchingReturns(cfg)
        model.sample(50, 24, make_rng(1))
        assert model.regime_indices is not None
        assert model.regime_indices.shape == (50, 24)
        # All values should be valid regime indices
        assert np.all(model.regime_indices >= 0)
        assert np.all(model.regime_indices < 2)

    def test_single_regime_close_to_mvn(self) -> None:
        """With 1 absorbing state, should behave like standard MVN."""
        cfg = RegimeSwitchingConfig(
            regimes=[
                RegimeConfig(
                    name="only",
                    expected_annual_returns=[0.07, 0.03],
                    annual_volatilities=[0.16, 0.06],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.03,
                    inflation_vol=0.01,
                ),
                RegimeConfig(
                    name="unused",
                    expected_annual_returns=[0.0, 0.0],
                    annual_volatilities=[0.16, 0.06],
                    correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    inflation_mean=0.03,
                    inflation_vol=0.01,
                ),
            ],
            transition_matrix=[[1.0, 0.0], [0.0, 1.0]],
            initial_regime=0,
        )
        model = RegimeSwitchingReturns(cfg)
        result = model.sample(5000, 120, make_rng(42))
        # Mean monthly return for stocks should be ~0.07/12
        mean_stock = result[:, :, 0].mean()
        assert abs(mean_stock - 0.07 / 12) < 0.001


class TestRegimeSwitchingInflation:
    def test_output_shape(self) -> None:
        cfg = _make_rs_config(2)
        rs_returns = RegimeSwitchingReturns(cfg)
        rs_returns.sample(100, 60, make_rng(42))
        assert rs_returns.regime_indices is not None

        infl_model = RegimeSwitchingInflationModel(regimes=list(cfg.regimes))
        rates = infl_model.sample(100, 60, make_rng(99), rs_returns.regime_indices)
        assert rates.shape == (100, 60)

    def test_determinism(self) -> None:
        cfg = _make_rs_config(2)
        rs = RegimeSwitchingReturns(cfg)
        rs.sample(50, 24, make_rng(42))
        assert rs.regime_indices is not None

        m1 = RegimeSwitchingInflationModel(regimes=list(cfg.regimes))
        m2 = RegimeSwitchingInflationModel(regimes=list(cfg.regimes))
        r1 = m1.sample(50, 24, make_rng(7), rs.regime_indices)
        r2 = m2.sample(50, 24, make_rng(7), rs.regime_indices)
        np.testing.assert_array_equal(r1, r2)


class TestEngineWithRegimeSwitching:
    def _make_rs_market(self) -> MarketAssumptions:
        return MarketAssumptions(
            asset_allocations=[
                AssetAllocation(
                    assets=[
                        AssetClass(name="US Stocks", weight=0.7),
                        AssetClass(name="US Bonds", weight=0.3),
                    ]
                )
            ],
            expected_annual_returns=[0.07, 0.03],
            annual_volatilities=[0.16, 0.06],
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
            inflation_mean=0.03,
            inflation_vol=0.01,
            return_model="regime_switching",
            regime_switching=_make_rs_config(3),
        )

    def test_simulate_runs(self) -> None:
        plan = default_plan()
        market = self._make_rs_market()
        result = simulate(plan, market, default_policies(), SimulationConfig(n_paths=100, seed=42))
        assert 0.0 <= result.success_probability <= 1.0

    def test_determinism(self) -> None:
        plan = default_plan()
        market = self._make_rs_market()
        sim = SimulationConfig(n_paths=200, seed=42)
        r1 = simulate(plan, market, default_policies(), sim)
        r2 = simulate(plan, market, default_policies(), sim)
        assert r1.success_probability == r2.success_probability

    def test_missing_config_raises(self) -> None:
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
            expected_annual_returns=[0.07],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
            return_model="regime_switching",
        )
        with pytest.raises(ValueError, match="regime_switching config"):
            simulate(
                default_plan(),
                market,
                default_policies(),
                SimulationConfig(n_paths=10, seed=1),
            )
