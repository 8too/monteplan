"""Tests for the simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.config.defaults import (
    default_market,
    default_plan,
    default_policies,
)
from monteplan.config.schema import (
    AccountConfig,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    RegimeConfig,
    RegimeSwitchingConfig,
    SimulationConfig,
    SpendingPolicyConfig,
)
from monteplan.core.engine import simulate


class TestSimulateBasic:
    def test_output_shape(self) -> None:
        plan = default_plan()
        sim = SimulationConfig(n_paths=100, seed=42)
        result = simulate(plan, default_market(), default_policies(), sim)
        assert result.n_paths == 100
        assert result.n_steps == (plan.end_age - plan.current_age) * 12

    def test_success_probability_range(self) -> None:
        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=500, seed=42),
        )
        assert 0.0 <= result.success_probability <= 1.0

    def test_terminal_wealth_percentiles_ordered(self) -> None:
        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=500, seed=42),
        )
        pcts = result.terminal_wealth_percentiles
        assert pcts["p5"] <= pcts["p25"] <= pcts["p50"] <= pcts["p75"] <= pcts["p95"]

    def test_wealth_time_series_keys(self) -> None:
        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=100, seed=42),
        )
        expected_keys = {"p5", "p25", "p50", "p75", "p95", "mean"}
        assert set(result.wealth_time_series.keys()) == expected_keys

    def test_wealth_time_series_length(self) -> None:
        plan = default_plan()
        sim = SimulationConfig(n_paths=100, seed=42)
        result = simulate(plan, default_market(), default_policies(), sim)
        n_steps = (plan.end_age - plan.current_age) * 12
        for ts in result.wealth_time_series.values():
            assert len(ts) == n_steps + 1  # includes t=0

    def test_store_paths_flag(self) -> None:
        result_no = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=50, seed=42, store_paths=False),
        )
        assert result_no.all_paths is None

        result_yes = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=50, seed=42, store_paths=True),
        )
        assert result_yes.all_paths is not None
        assert result_yes.all_paths.shape == (50, result_yes.n_steps + 1)
        assert result_yes.inflation_rates.shape == (50, result_yes.n_steps)
        assert result_yes.spending_history.shape == (50, result_yes.n_steps)


class TestDeterminism:
    def test_same_seed_same_results(self) -> None:
        """Two runs with the same seed must produce identical results."""
        cfg = SimulationConfig(n_paths=200, seed=42)
        r1 = simulate(default_plan(), default_market(), default_policies(), cfg)
        r2 = simulate(default_plan(), default_market(), default_policies(), cfg)
        assert r1.success_probability == r2.success_probability
        np.testing.assert_array_equal(
            r1.wealth_time_series["p50"],
            r2.wealth_time_series["p50"],
        )

    def test_different_seed_different_results(self) -> None:
        cfg1 = SimulationConfig(n_paths=200, seed=1)
        cfg2 = SimulationConfig(n_paths=200, seed=2)
        r1 = simulate(default_plan(), default_market(), default_policies(), cfg1)
        r2 = simulate(default_plan(), default_market(), default_policies(), cfg2)
        # Extremely unlikely to be exactly equal
        assert r1.success_probability != r2.success_probability


class TestGolden:
    """Golden test: fixed seed → fixed numeric output.

    If this test breaks, either the engine logic changed (intentionally)
    or a regression was introduced. Update the golden values deliberately.
    """

    def test_golden_basic_retirement(self) -> None:
        plan = PlanConfig(
            current_age=30,
            retirement_age=65,
            end_age=95,
            accounts=[
                AccountConfig(account_type="taxable", balance=50_000, annual_contribution=6_000),
                AccountConfig(
                    account_type="traditional", balance=100_000, annual_contribution=20_000
                ),
                AccountConfig(account_type="roth", balance=30_000, annual_contribution=7_000),
            ],
            monthly_income=8_000,
            monthly_spending=5_000,
        )
        market = MarketAssumptions(
            assets=[
                AssetClass(name="US Stocks", weight=0.7),
                AssetClass(name="US Bonds", weight=0.3),
            ],
            expected_annual_returns=[0.07, 0.03],
            annual_volatilities=[0.16, 0.06],
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
            inflation_mean=0.03,
            inflation_vol=0.01,
        )
        policies = PolicyBundle()
        sim = SimulationConfig(n_paths=5000, seed=42)

        result = simulate(plan, market, policies, sim)

        # Golden values — update these if engine logic changes
        # With constant real spending and 3% inflation, many paths deplete
        # over 30 years of retirement. ~48% success is expected.
        assert result.success_probability == pytest.approx(0.4794, abs=0.03)
        assert result.terminal_wealth_percentiles["p75"] > 1_000_000
        # Median wealth at retirement should be substantial
        assert result.wealth_time_series["p50"][420] > 3_000_000

    def test_golden_regime_switching(self) -> None:
        """Golden test for regime-switching return model."""
        plan = default_plan()
        rs_config = RegimeSwitchingConfig(
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
        market = MarketAssumptions(
            assets=[
                AssetClass(name="US Stocks", weight=0.7),
                AssetClass(name="US Bonds", weight=0.3),
            ],
            expected_annual_returns=[0.07, 0.03],
            annual_volatilities=[0.16, 0.06],
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
            inflation_mean=0.03,
            inflation_vol=0.01,
            return_model="regime_switching",
            regime_switching=rs_config,
        )
        result = simulate(plan, market, PolicyBundle(), SimulationConfig(n_paths=5000, seed=42))
        assert result.success_probability == pytest.approx(0.3282, abs=0.03)

    def test_golden_antithetic(self) -> None:
        """Golden test for antithetic variates."""
        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=5000, seed=42, antithetic=True),
        )
        assert result.success_probability == pytest.approx(0.7658, abs=0.03)
        assert result.n_paths == 5000


class TestPercentOfPortfolioPolicy:
    def test_percent_policy_runs(self) -> None:
        plan = default_plan()
        policies = PolicyBundle(
            spending=SpendingPolicyConfig(
                policy_type="percent_of_portfolio",
                withdrawal_rate=0.04,
            ),
        )
        result = simulate(plan, default_market(), policies, SimulationConfig(n_paths=100, seed=42))
        assert 0.0 <= result.success_probability <= 1.0


def test_benchmark_simulate(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Benchmark: simulate() should complete in < 2s for 5000 paths × 780 steps."""
    plan = default_plan()
    market = default_market()
    policies = default_policies()
    sim = SimulationConfig(n_paths=5000, seed=42)

    result = benchmark(simulate, plan, market, policies, sim)
    assert result.success_probability >= 0


def test_benchmark_us_federal(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Benchmark: simulate() with US federal tax model."""
    plan = default_plan()
    market = default_market()
    policies = PolicyBundle(tax_model="us_federal")
    sim = SimulationConfig(n_paths=5000, seed=42)

    result = benchmark(simulate, plan, market, policies, sim)
    assert result.success_probability >= 0


def test_benchmark_regime_switching(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Benchmark: simulate() with regime-switching return model."""
    plan = default_plan()
    rs_config = RegimeSwitchingConfig(
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
    market = MarketAssumptions(
        assets=[
            AssetClass(name="US Stocks", weight=0.7),
            AssetClass(name="US Bonds", weight=0.3),
        ],
        expected_annual_returns=[0.07, 0.03],
        annual_volatilities=[0.16, 0.06],
        correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        inflation_mean=0.03,
        inflation_vol=0.01,
        return_model="regime_switching",
        regime_switching=rs_config,
    )
    sim = SimulationConfig(n_paths=5000, seed=42)
    result = benchmark(simulate, plan, market, PolicyBundle(), sim)
    assert result.success_probability >= 0
