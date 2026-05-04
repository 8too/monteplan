"""Tests for 6-asset global diversification (v0.6)."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.analytics.sensitivity import _build_param_registry, run_sensitivity
from monteplan.config.defaults import (
    ALL_ASSET_NAMES,
    BOND_ASSET_NAMES,
    CORRELATION_CONSENSUS,
    CORRELATION_CRISIS_AWARE,
    STOCK_ASSET_NAMES,
    build_global_weights,
    default_market,
    default_plan,
    default_policies,
    global_market,
    us_only_market,
)
from monteplan.config.schema import (
    GlidePath,
    SimulationConfig,
)
from monteplan.core.engine import simulate


class TestDefaultMarket:
    def test_default_market_has_6_assets(self) -> None:
        market = default_market()
        assert len(market.asset_allocations[0].assets) == 6

    def test_asset_names(self) -> None:
        market = default_market()
        names = [a.name for a in market.asset_allocations[0].assets]
        assert names == ALL_ASSET_NAMES

    def test_weights_sum_to_one(self) -> None:
        market = default_market()
        total = sum(a.weight for a in market.asset_allocations[0].assets)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_stock_bond_split_is_60_40(self) -> None:
        market = default_market()
        stock_total = sum(
            a.weight for a in market.asset_allocations[0].assets if "Stock" in a.name
        )
        bond_total = sum(a.weight for a in market.asset_allocations[0].assets if "Bond" in a.name)
        assert stock_total == pytest.approx(0.60, abs=1e-9)
        assert bond_total == pytest.approx(0.40, abs=1e-9)

    def test_regional_split_within_stocks(self) -> None:
        market = default_market()
        stock_weights = [a.weight for a in market.asset_allocations[0].assets if "Stock" in a.name]
        total = sum(stock_weights)
        ratios = [w / total for w in stock_weights]
        assert ratios == pytest.approx([0.60, 0.30, 0.10], abs=1e-9)

    def test_regional_split_within_bonds(self) -> None:
        market = default_market()
        bond_weights = [a.weight for a in market.asset_allocations[0].assets if "Bond" in a.name]
        total = sum(bond_weights)
        ratios = [w / total for w in bond_weights]
        assert ratios == pytest.approx([0.60, 0.30, 0.10], abs=1e-9)

    def test_6_returns_and_vols(self) -> None:
        market = default_market()
        assert len(market.expected_annual_returns) == 6
        assert len(market.annual_volatilities) == 6

    def test_6x6_correlation_matrix(self) -> None:
        market = default_market()
        assert len(market.correlation_matrix) == 6
        for row in market.correlation_matrix:
            assert len(row) == 6

    def test_bond_type_aggregate_vs_treasuries(self) -> None:
        agg = default_market(bond_type="aggregate")
        tre = default_market(bond_type="treasuries")
        # US Bond return differs
        assert agg.expected_annual_returns[3] == 0.05
        assert tre.expected_annual_returns[3] == 0.045
        # US Bond vol differs
        assert agg.annual_volatilities[3] == 0.07
        assert tre.annual_volatilities[3] == 0.06


class TestCorrelationMatrices:
    def test_consensus_is_valid_psd(self) -> None:
        corr = np.array(CORRELATION_CONSENSUS)
        assert corr.shape == (6, 6)
        # Symmetric
        np.testing.assert_array_almost_equal(corr, corr.T)
        # Diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(corr), 1.0)
        # Positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(corr)
        assert np.all(eigenvalues > -1e-10)

    def test_crisis_aware_is_valid_psd(self) -> None:
        corr = np.array(CORRELATION_CRISIS_AWARE)
        assert corr.shape == (6, 6)
        np.testing.assert_array_almost_equal(corr, corr.T)
        np.testing.assert_array_almost_equal(np.diag(corr), 1.0)
        eigenvalues = np.linalg.eigvalsh(corr)
        assert np.all(eigenvalues > -1e-10)


class TestBuildGlobalWeights:
    def test_default_weights(self) -> None:
        weights = build_global_weights()
        assert len(weights) == 6
        assert sum(weights) == pytest.approx(1.0, abs=1e-9)
        # 60% stocks: 36% US, 18% ExUS, 6% EM
        assert weights[0] == pytest.approx(0.36, abs=1e-9)
        assert weights[1] == pytest.approx(0.18, abs=1e-9)
        assert weights[2] == pytest.approx(0.06, abs=1e-9)
        # 40% bonds: 24% US, 12% ExUS, 4% EM
        assert weights[3] == pytest.approx(0.24, abs=1e-9)
        assert weights[4] == pytest.approx(0.12, abs=1e-9)
        assert weights[5] == pytest.approx(0.04, abs=1e-9)

    def test_custom_stock_pct(self) -> None:
        weights = build_global_weights(stock_pct=0.80)
        stock_total = sum(weights[:3])
        bond_total = sum(weights[3:])
        assert stock_total == pytest.approx(0.80, abs=1e-9)
        assert bond_total == pytest.approx(0.20, abs=1e-9)

    def test_custom_regional_split(self) -> None:
        weights = build_global_weights(
            stock_pct=0.50,
            stock_regional=[0.70, 0.20, 0.10],
            bond_regional=[0.80, 0.15, 0.05],
        )
        assert weights[0] == pytest.approx(0.35, abs=1e-9)  # US stocks: 0.5 * 0.7
        assert weights[3] == pytest.approx(0.40, abs=1e-9)  # US bonds: 0.5 * 0.8

    def test_bond_regional_defaults_to_stock(self) -> None:
        weights = build_global_weights(
            stock_pct=0.60,
            stock_regional=[0.50, 0.30, 0.20],
        )
        # Bond regional should match stock regional
        stock_ratios = [w / 0.60 for w in weights[:3]]
        bond_ratios = [w / 0.40 for w in weights[3:]]
        assert stock_ratios == pytest.approx(bond_ratios, abs=1e-9)


class TestUSOnlyMarket:
    def test_us_only_market_has_2_assets(self) -> None:
        market = us_only_market()
        assert len(market.asset_allocations[0].assets) == 2
        assert market.asset_allocations[0].assets[0].name == "US Stocks"
        assert market.asset_allocations[0].assets[1].name == "US Bonds"

    def test_us_only_weights(self) -> None:
        market = us_only_market()
        assert market.asset_allocations[0].assets[0].weight == pytest.approx(0.70)
        assert market.asset_allocations[0].assets[1].weight == pytest.approx(0.30)

    def test_us_only_aggregate_vs_treasuries(self) -> None:
        agg = us_only_market(bond_type="aggregate")
        tre = us_only_market(bond_type="treasuries")
        assert agg.expected_annual_returns[1] == 0.05
        assert tre.expected_annual_returns[1] == 0.045


class TestGlobalMarketAlias:
    def test_global_market_equals_default(self) -> None:
        gm = global_market()
        dm = default_market()
        assert gm.asset_allocations[0].assets == dm.asset_allocations[0].assets
        assert gm.expected_annual_returns == dm.expected_annual_returns


class TestSimulationWith6Assets:
    def test_6_asset_simulation_runs(self) -> None:
        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=100, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0
        assert result.n_paths == 100

    def test_2_asset_still_works(self) -> None:
        result = simulate(
            default_plan(),
            us_only_market(),
            default_policies(),
            SimulationConfig(n_paths=100, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0

    def test_6_asset_with_glide_path(self) -> None:
        market = default_market()
        weights = [a.weight for a in market.asset_allocations[0].assets]
        # End weights: 30% stocks, 70% bonds with same regional ratios
        end_weights = build_global_weights(stock_pct=0.30)

        market_gp = market.model_copy(
            update={
                "glide_path": GlidePath(
                    start_age=30,
                    start_weights=weights,
                    end_age=70,
                    end_weights=end_weights,
                ),
            }
        )
        result = simulate(
            default_plan(),
            market_gp,
            default_policies(),
            SimulationConfig(n_paths=100, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0


class TestEquityAllocationSetter:
    def test_equity_allocation_setter_preserves_ratios(self) -> None:
        market = default_market()
        registry = _build_param_registry(default_plan(), market)
        assert "Equity Allocation" in registry

        spec = registry["Equity Allocation"]
        # Current equity allocation should be 0.60
        assert spec.getter(market) == pytest.approx(0.60, abs=1e-9)

        # Set to 80% equities
        new_market = spec.setter(market, 0.80)
        stock_total = sum(
            a.weight for a in new_market.asset_allocations[0].assets if "Stock" in a.name
        )
        bond_total = sum(
            a.weight for a in new_market.asset_allocations[0].assets if "Bond" in a.name
        )
        assert stock_total == pytest.approx(0.80, abs=1e-9)
        assert bond_total == pytest.approx(0.20, abs=1e-9)

        # Regional ratios within stocks should be preserved
        stock_weights = [
            a.weight for a in new_market.asset_allocations[0].assets if "Stock" in a.name
        ]
        ratios = [w / stock_total for w in stock_weights]
        assert ratios == pytest.approx([0.60, 0.30, 0.10], abs=1e-9)

    def test_equity_allocation_setter_2_asset(self) -> None:
        """Backward compat: works with 2-asset US-only market."""
        market = us_only_market()
        registry = _build_param_registry(default_plan(), market)
        assert "Equity Allocation" in registry

        spec = registry["Equity Allocation"]
        assert spec.getter(market) == pytest.approx(0.70, abs=1e-9)

        new_market = spec.setter(market, 0.50)
        assert new_market.asset_allocations[0].assets[0].weight == pytest.approx(0.50, abs=1e-9)
        assert new_market.asset_allocations[0].assets[1].weight == pytest.approx(0.50, abs=1e-9)

    def test_sensitivity_with_6_assets(self) -> None:
        report = run_sensitivity(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=100, seed=42),
            parameters=["Equity Allocation", "US Stocks Return"],
            max_workers=1,
        )
        assert len(report.results) == 2
        names = {r.parameter_name for r in report.results}
        assert names == {"Equity Allocation", "US Stocks Return"}


class TestAssetNameConstants:
    def test_stock_names(self) -> None:
        assert STOCK_ASSET_NAMES == ["US Stocks", "Ex-US Dev Stocks", "EM Stocks"]

    def test_bond_names(self) -> None:
        assert BOND_ASSET_NAMES == ["US Bonds", "Ex-US Dev Bonds", "EM Bonds"]

    def test_all_names(self) -> None:
        assert ALL_ASSET_NAMES == STOCK_ASSET_NAMES + BOND_ASSET_NAMES
        assert len(ALL_ASSET_NAMES) == 6
