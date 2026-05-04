"""Tests for multivariate normal return model."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import PCG64DXSM, Generator

from monteplan.config.schema import AssetAllocation, AssetClass, MarketAssumptions
from monteplan.models.returns.mvn import MultivariateNormalReturns, StudentTReturns


@pytest.fixture
def market() -> MarketAssumptions:
    return MarketAssumptions(
        asset_allocations=[
            AssetAllocation(
                assets=[
                    AssetClass(name="Stocks", weight=0.6),
                    AssetClass(name="Bonds", weight=0.4),
                ]
            )
        ],
        expected_annual_returns=[0.08, 0.03],
        annual_volatilities=[0.16, 0.05],
        correlation_matrix=[[1.0, 0.2], [0.2, 1.0]],
    )


class TestMVN:
    def test_output_shape(self, market: MarketAssumptions, rng) -> None:  # type: ignore[no-untyped-def]
        model = MultivariateNormalReturns(market)
        result = model.sample(100, 12, rng)
        assert result.shape == (100, 12, 2)

    def test_sample_means_close_to_expected(self, market: MarketAssumptions) -> None:
        """With 100k samples, means should be within 1% of expected."""
        rng = Generator(PCG64DXSM(123))
        model = MultivariateNormalReturns(market)
        # 100k paths, 1 step
        result = model.sample(100_000, 1, rng)
        sample_means = result[:, 0, :].mean(axis=0)
        expected_monthly = np.array([0.08, 0.03]) / 12.0
        np.testing.assert_allclose(sample_means, expected_monthly, atol=0.001)

    def test_sample_vols_close_to_expected(self, market: MarketAssumptions) -> None:
        """With 100k samples, vols should be within 2% of expected."""
        rng = Generator(PCG64DXSM(456))
        model = MultivariateNormalReturns(market)
        result = model.sample(100_000, 1, rng)
        sample_stds = result[:, 0, :].std(axis=0)
        expected_monthly_vols = np.array([0.16, 0.05]) / np.sqrt(12.0)
        np.testing.assert_allclose(sample_stds, expected_monthly_vols, rtol=0.02)

    def test_correlation_close_to_expected(self, market: MarketAssumptions) -> None:
        """With 100k samples, correlation should be within 0.05."""
        rng = Generator(PCG64DXSM(789))
        model = MultivariateNormalReturns(market)
        result = model.sample(100_000, 1, rng)
        sample_corr = np.corrcoef(result[:, 0, 0], result[:, 0, 1])[0, 1]
        assert abs(sample_corr - 0.2) < 0.05

    def test_deterministic_with_same_seed(self, market: MarketAssumptions) -> None:
        model = MultivariateNormalReturns(market)
        r1 = model.sample(10, 5, Generator(PCG64DXSM(42)))
        r2 = model.sample(10, 5, Generator(PCG64DXSM(42)))
        np.testing.assert_array_equal(r1, r2)


class TestStudentT:
    def test_output_shape(self) -> None:
        market = MarketAssumptions(
            asset_allocations=[
                AssetAllocation(
                    assets=[AssetClass(name="S", weight=0.6), AssetClass(name="B", weight=0.4)]
                )
            ],
            expected_annual_returns=[0.08, 0.03],
            annual_volatilities=[0.16, 0.05],
            correlation_matrix=[[1.0, 0.2], [0.2, 1.0]],
            return_model="student_t",
            degrees_of_freedom=5.0,
        )
        model = StudentTReturns(market)
        result = model.sample(100, 12, Generator(PCG64DXSM(42)))
        assert result.shape == (100, 12, 2)

    def test_fatter_tails_than_normal(self) -> None:
        """Student-t should produce more extreme returns than normal."""
        market_mvn = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="S", weight=1.0)])],
            expected_annual_returns=[0.08],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
        )
        market_t = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="S", weight=1.0)])],
            expected_annual_returns=[0.08],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
            return_model="student_t",
            degrees_of_freedom=4.0,
        )
        mvn = MultivariateNormalReturns(market_mvn)
        t_model = StudentTReturns(market_t)

        mvn_samples = mvn.sample(100_000, 1, Generator(PCG64DXSM(42)))
        t_samples = t_model.sample(100_000, 1, Generator(PCG64DXSM(42)))

        # Student-t should have higher kurtosis (fatter tails)
        from scipy.stats import kurtosis

        mvn_kurt = kurtosis(mvn_samples[:, 0, 0])
        t_kurt = kurtosis(t_samples[:, 0, 0])
        assert t_kurt > mvn_kurt

    def test_deterministic(self) -> None:
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="S", weight=1.0)])],
            expected_annual_returns=[0.08],
            annual_volatilities=[0.16],
            correlation_matrix=[[1.0]],
            return_model="student_t",
            degrees_of_freedom=5.0,
        )
        model = StudentTReturns(market)
        r1 = model.sample(10, 5, Generator(PCG64DXSM(42)))
        r2 = model.sample(10, 5, Generator(PCG64DXSM(42)))
        np.testing.assert_array_equal(r1, r2)
