"""Tests for stress scenario overlays."""

from __future__ import annotations

import numpy as np
import pytest

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
    StressScenario,
)
from monteplan.core.engine import simulate
from monteplan.core.timeline import Timeline
from monteplan.models.stress import apply_stress_scenarios


@pytest.fixture
def timeline() -> Timeline:
    return Timeline.from_ages(30, 65, 95)


class TestApplyStressScenarios:
    def test_crash_modifies_returns(self, timeline: Timeline) -> None:
        """Crash scenario should set returns to negative then recovery."""
        n_paths, n_steps, n_assets = 10, timeline.n_steps, 2
        returns = np.full((n_paths, n_steps, n_assets), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenario = StressScenario(
            name="crash",
            scenario_type="crash",
            start_age=40,
            duration_months=24,
            severity=1.0,
        )
        returns, inflation = apply_stress_scenarios(
            returns,
            inflation,
            [scenario],
            timeline,
        )

        start_step = int(round((40 - 30) * 12))
        # Crash period returns should be negative (first half)
        crash_half = returns[:, start_step : start_step + 12, :]
        assert np.all(crash_half < 0), "Crash decline phase should have negative returns"

        # Recovery phase should be positive
        recovery_half = returns[:, start_step + 12 : start_step + 24, :]
        assert np.all(recovery_half > 0), "Crash recovery phase should have positive returns"

        # Outside the crash window, returns should be unchanged
        assert np.allclose(returns[:, 0, :], 0.005)

    def test_lost_decade_near_zero(self, timeline: Timeline) -> None:
        """Lost decade should set returns near zero."""
        n_paths, n_steps, n_assets = 10, timeline.n_steps, 1
        returns = np.full((n_paths, n_steps, n_assets), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenario = StressScenario(
            name="lost_decade",
            scenario_type="lost_decade",
            start_age=35,
            duration_months=120,
            severity=1.0,
        )
        returns, _ = apply_stress_scenarios(
            returns,
            inflation,
            [scenario],
            timeline,
        )

        start_step = int(round((35 - 30) * 12))
        affected = returns[:, start_step : start_step + 120, :]
        # At severity=1.0, returns should be 0.001*(1-1.0) = 0.0
        assert np.allclose(affected, 0.0)

    def test_high_inflation(self, timeline: Timeline) -> None:
        """High inflation should raise inflation rates in the affected window."""
        n_paths, n_steps = 10, timeline.n_steps
        returns = np.full((n_paths, n_steps, 1), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenario = StressScenario(
            name="high_inflation",
            scenario_type="high_inflation",
            start_age=50,
            duration_months=60,
            severity=1.0,
        )
        _, inflation = apply_stress_scenarios(
            returns,
            inflation,
            [scenario],
            timeline,
        )

        start_step = int(round((50 - 30) * 12))
        affected = inflation[:, start_step : start_step + 60]
        # At severity=1.0: target_annual = 0.06 + 0.02*1.0 = 0.08, monthly = 0.08/12
        expected_monthly = 0.08 / 12.0
        assert np.allclose(affected, expected_monthly)

        # Unaffected region should still be 0.002
        assert np.allclose(inflation[:, 0], 0.002)

    def test_sequence_risk(self, timeline: Timeline) -> None:
        """Sequence risk should produce bad returns early, recovery later."""
        n_paths, n_steps, n_assets = 10, timeline.n_steps, 2
        returns = np.full((n_paths, n_steps, n_assets), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenario = StressScenario(
            name="seq_risk",
            scenario_type="sequence_risk",
            start_age=65,
            duration_months=120,
            severity=1.0,
        )
        returns, _ = apply_stress_scenarios(
            returns,
            inflation,
            [scenario],
            timeline,
        )

        start_step = int(round((65 - 30) * 12))
        # First 60 months: bad returns (-0.02 at severity=1.0)
        bad = returns[:, start_step : start_step + 60, :]
        assert np.allclose(bad, -0.02)

        # Next 60 months: above-average (0.01)
        good = returns[:, start_step + 60 : start_step + 120, :]
        assert np.allclose(good, 0.01)

    def test_scenario_before_timeline_clipped(self, timeline: Timeline) -> None:
        """Scenario starting before current age should be clipped."""
        n_paths, n_steps = 5, timeline.n_steps
        returns = np.full((n_paths, n_steps, 1), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenario = StressScenario(
            name="early",
            scenario_type="lost_decade",
            start_age=25,  # Before current_age=30
            duration_months=120,  # Extends to age 35
            severity=1.0,
        )
        returns, _ = apply_stress_scenarios(
            returns,
            inflation,
            [scenario],
            timeline,
        )

        # Should affect steps 0 to 60 (age 25+120mo = age 35, but clipped at age 30 start)
        # start_step = (25-30)*12 = -60, clipped to 0
        # end_step = -60 + 120 = 60
        affected = returns[:, 0:60, :]
        assert np.allclose(affected, 0.0)  # lost decade at severity=1.0

        # After step 60, should be original
        assert np.allclose(returns[:, 60, :], 0.005)

    def test_scenario_past_timeline_clipped(self, timeline: Timeline) -> None:
        """Scenario extending past end should be clipped."""
        n_paths, n_steps = 5, timeline.n_steps
        returns = np.full((n_paths, n_steps, 1), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenario = StressScenario(
            name="late",
            scenario_type="lost_decade",
            start_age=90,
            duration_months=120,  # Would extend to age 100, past end_age=95
            severity=1.0,
        )
        returns, _ = apply_stress_scenarios(
            returns,
            inflation,
            [scenario],
            timeline,
        )

        start_step = int(round((90 - 30) * 12))
        # Should affect from start_step to end of timeline
        affected = returns[:, start_step:, :]
        assert np.allclose(affected, 0.0)

    def test_no_scenarios_is_noop(self, timeline: Timeline) -> None:
        """Empty scenario list should not modify arrays."""
        returns = np.full((5, timeline.n_steps, 1), 0.005)
        inflation = np.full((5, timeline.n_steps), 0.002)
        original_returns = returns.copy()
        original_inflation = inflation.copy()

        apply_stress_scenarios(returns, inflation, [], timeline)

        np.testing.assert_array_equal(returns, original_returns)
        np.testing.assert_array_equal(inflation, original_inflation)

    def test_multiple_scenarios(self, timeline: Timeline) -> None:
        """Multiple scenarios should all be applied."""
        n_paths, n_steps = 5, timeline.n_steps
        returns = np.full((n_paths, n_steps, 1), 0.005)
        inflation = np.full((n_paths, n_steps), 0.002)

        scenarios = [
            StressScenario(
                name="crash",
                scenario_type="crash",
                start_age=40,
                duration_months=24,
                severity=1.0,
            ),
            StressScenario(
                name="inflation",
                scenario_type="high_inflation",
                start_age=50,
                duration_months=60,
                severity=0.5,
            ),
        ]
        returns, inflation = apply_stress_scenarios(
            returns,
            inflation,
            scenarios,
            timeline,
        )

        # Crash should have modified returns around age 40
        crash_start = int(round((40 - 30) * 12))
        assert not np.allclose(returns[:, crash_start, :], 0.005)

        # High inflation should have modified inflation around age 50
        infl_start = int(round((50 - 30) * 12))
        assert not np.allclose(inflation[:, infl_start], 0.002)

    def test_severity_scales(self, timeline: Timeline) -> None:
        """Higher severity should produce larger effects."""
        n_paths, n_steps = 5, timeline.n_steps

        returns_lo = np.full((n_paths, n_steps, 1), 0.005)
        inflation_lo = np.full((n_paths, n_steps), 0.002)
        returns_hi = np.full((n_paths, n_steps, 1), 0.005)
        inflation_hi = np.full((n_paths, n_steps), 0.002)

        scenario_lo = StressScenario(
            name="mild",
            scenario_type="crash",
            start_age=40,
            duration_months=24,
            severity=0.5,
        )
        scenario_hi = StressScenario(
            name="severe",
            scenario_type="crash",
            start_age=40,
            duration_months=24,
            severity=1.5,
        )

        apply_stress_scenarios(returns_lo, inflation_lo, [scenario_lo], timeline)
        apply_stress_scenarios(returns_hi, inflation_hi, [scenario_hi], timeline)

        crash_start = int(round((40 - 30) * 12))
        # Higher severity should produce more negative crash returns
        lo_min = returns_lo[:, crash_start : crash_start + 12, :].min()
        hi_min = returns_hi[:, crash_start : crash_start + 12, :].min()
        assert hi_min < lo_min


class TestEngineIntegration:
    def test_crash_reduces_success(self) -> None:
        """A crash scenario should reduce success probability."""
        plan = PlanConfig(
            current_age=60,
            retirement_age=65,
            end_age=90,
            accounts=[AccountConfig(balance=1_000_000)],
            monthly_income=5_000,
            monthly_spending=4_000,
        )
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="Stocks", weight=1.0)])],
            expected_annual_returns=[0.07],
            annual_volatilities=[0.15],
            correlation_matrix=[[1.0]],
        )

        # Without stress
        result_no_stress = simulate(
            plan,
            market,
            PolicyBundle(),
            SimulationConfig(n_paths=500, seed=42),
        )

        # With crash at retirement
        crash = StressScenario(
            name="retirement_crash",
            scenario_type="crash",
            start_age=65,
            duration_months=24,
            severity=1.5,
        )
        result_stress = simulate(
            plan,
            market,
            PolicyBundle(),
            SimulationConfig(n_paths=500, seed=42, stress_scenarios=[crash]),
        )

        assert result_stress.success_probability <= result_no_stress.success_probability

    def test_stress_scenario_deterministic(self) -> None:
        """Same config with stress should produce same results."""
        plan = PlanConfig(
            current_age=60,
            retirement_age=65,
            end_age=80,
            accounts=[AccountConfig(balance=500_000)],
            monthly_income=3_000,
            monthly_spending=2_500,
        )
        market = MarketAssumptions(
            asset_allocations=[AssetAllocation(assets=[AssetClass(name="S", weight=1.0)])],
            expected_annual_returns=[0.06],
            annual_volatilities=[0.12],
            correlation_matrix=[[1.0]],
        )
        stress = StressScenario(
            name="crash",
            scenario_type="crash",
            start_age=65,
            duration_months=12,
            severity=1.0,
        )
        cfg = SimulationConfig(n_paths=100, seed=42, stress_scenarios=[stress])

        r1 = simulate(plan, market, PolicyBundle(), cfg)
        r2 = simulate(plan, market, PolicyBundle(), cfg)
        assert r1.success_probability == r2.success_probability
