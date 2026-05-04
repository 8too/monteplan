"""Tests for antithetic variates variance reduction."""

from __future__ import annotations

import numpy as np

from monteplan.config.defaults import (
    default_market,
    default_plan,
    default_policies,
)
from monteplan.config.schema import (
    SimulationConfig,
)
from monteplan.core.engine import simulate
from monteplan.core.rng import make_rng
from monteplan.models.inflation import OUInflationModel
from monteplan.models.returns.mvn import MultivariateNormalReturns, StudentTReturns


class TestAntitheticMVN:
    def test_output_shape(self) -> None:
        market = default_market()
        n_assets = len(market.asset_allocations[0].assets)
        model = MultivariateNormalReturns(market, antithetic=True)
        result = model.sample(100, 60, make_rng(42))
        assert result.shape == (100, 60, n_assets)

    def test_determinism(self) -> None:
        market = default_market()
        m1 = MultivariateNormalReturns(market, antithetic=True)
        m2 = MultivariateNormalReturns(market, antithetic=True)
        r1 = m1.sample(200, 120, make_rng(42))
        r2 = m2.sample(200, 120, make_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_no_bias(self) -> None:
        """Antithetic mean should be close to non-antithetic mean."""
        market = default_market()
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        m_normal = MultivariateNormalReturns(market, antithetic=False)
        m_anti = MultivariateNormalReturns(market, antithetic=True)

        # Use large sample for mean convergence
        r_normal = m_normal.sample(10000, 120, rng1)
        r_anti = m_anti.sample(10000, 120, rng2)

        # Means should be similar (not identical due to different draw sequences)
        np.testing.assert_allclose(
            r_normal.mean(axis=(0, 1)),
            r_anti.mean(axis=(0, 1)),
            atol=0.001,
        )


class TestAntitheticStudentT:
    def test_output_shape(self) -> None:
        market = default_market().model_copy(
            update={
                "return_model": "student_t",
                "degrees_of_freedom": 5.0,
            }
        )
        n_assets = len(market.asset_allocations[0].assets)
        model = StudentTReturns(market, antithetic=True)
        result = model.sample(100, 60, make_rng(42))
        assert result.shape == (100, 60, n_assets)

    def test_determinism(self) -> None:
        market = default_market().model_copy(
            update={
                "return_model": "student_t",
                "degrees_of_freedom": 5.0,
            }
        )
        m1 = StudentTReturns(market, antithetic=True)
        m2 = StudentTReturns(market, antithetic=True)
        r1 = m1.sample(200, 60, make_rng(42))
        r2 = m2.sample(200, 60, make_rng(42))
        np.testing.assert_array_equal(r1, r2)


class TestAntitheticInflation:
    def test_output_shape(self) -> None:
        model = OUInflationModel(antithetic=True)
        result = model.sample(100, 60, make_rng(42))
        assert result.shape == (100, 60)

    def test_determinism(self) -> None:
        m1 = OUInflationModel(antithetic=True)
        m2 = OUInflationModel(antithetic=True)
        r1 = m1.sample(200, 120, make_rng(42))
        r2 = m2.sample(200, 120, make_rng(42))
        np.testing.assert_array_equal(r1, r2)


class TestAntitheticEngine:
    def test_simulate_with_antithetic(self) -> None:
        result = simulate(
            default_plan(),
            default_market(),
            default_policies(),
            SimulationConfig(n_paths=200, seed=42, antithetic=True),
        )
        assert 0.0 <= result.success_probability <= 1.0
        # n_paths rounded to even
        assert result.n_paths % 2 == 0

    def test_determinism(self) -> None:
        sim = SimulationConfig(n_paths=200, seed=42, antithetic=True)
        r1 = simulate(default_plan(), default_market(), default_policies(), sim)
        r2 = simulate(default_plan(), default_market(), default_policies(), sim)
        assert r1.success_probability == r2.success_probability

    def test_odd_paths_rounded_up(self) -> None:
        sim = SimulationConfig(n_paths=101, seed=42, antithetic=True)
        result = simulate(default_plan(), default_market(), default_policies(), sim)
        # Engine should have rounded up to 102
        assert result.n_paths == 102


class TestPresets:
    def test_fast_preset(self) -> None:
        sim = SimulationConfig(preset="fast")
        assert sim.n_paths == 1000
        assert sim.antithetic is False

    def test_balanced_preset(self) -> None:
        sim = SimulationConfig(preset="balanced")
        assert sim.n_paths == 5000
        assert sim.antithetic is False

    def test_deep_preset(self) -> None:
        sim = SimulationConfig(preset="deep")
        assert sim.n_paths == 20000
        assert sim.antithetic is True

    def test_no_preset_defaults(self) -> None:
        sim = SimulationConfig()
        assert sim.n_paths == 5000
        assert sim.antithetic is False
        assert sim.preset is None
