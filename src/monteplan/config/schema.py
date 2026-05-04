"""Pydantic v2 configuration models for monteplan."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegimeConfig(BaseModel):
    """Parameters for a single market regime."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Regime name (e.g. 'bull', 'bear', 'normal')")
    expected_annual_returns: list[float] = Field(description="Per-asset annual returns")
    annual_volatilities: list[float] = Field(description="Per-asset annual volatilities")
    correlation_matrix: list[list[float]] = Field(description="Asset correlation matrix")
    inflation_mean: float = Field(description="Long-run annual inflation rate for this regime")
    inflation_vol: float = Field(ge=0, description="Inflation annual volatility for this regime")


class RegimeSwitchingConfig(BaseModel):
    """Markov regime-switching model configuration."""

    model_config = ConfigDict(extra="forbid")

    regimes: list[RegimeConfig] = Field(min_length=2, max_length=5)
    transition_matrix: list[list[float]] = Field(
        description="Row-stochastic transition matrix (rows sum to 1)"
    )
    initial_regime: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_regime_switching(self) -> RegimeSwitchingConfig:
        n = len(self.regimes)
        # Validate transition matrix dimensions
        if len(self.transition_matrix) != n:
            raise ValueError(
                f"transition_matrix must have {n} rows, got {len(self.transition_matrix)}"
            )
        for i, row in enumerate(self.transition_matrix):
            if len(row) != n:
                raise ValueError(f"transition_matrix row {i} has length {len(row)}, expected {n}")
            if abs(sum(row) - 1.0) > 1e-6:
                raise ValueError(f"transition_matrix row {i} sums to {sum(row)}, expected 1.0")
        # Validate initial_regime
        if self.initial_regime >= n:
            raise ValueError(
                f"initial_regime ({self.initial_regime}) must be < number of regimes ({n})"
            )
        # Validate all regimes have same number of assets
        n_assets = len(self.regimes[0].expected_annual_returns)
        for i, regime in enumerate(self.regimes):
            if len(regime.expected_annual_returns) != n_assets:
                raise ValueError(
                    f"Regime {i} has {len(regime.expected_annual_returns)} assets, "
                    f"expected {n_assets}"
                )
            if len(regime.annual_volatilities) != n_assets:
                raise ValueError(
                    f"Regime {i} volatilities length mismatch: "
                    f"{len(regime.annual_volatilities)} vs {n_assets}"
                )
            if len(regime.correlation_matrix) != n_assets:
                raise ValueError(f"Regime {i} correlation_matrix must be {n_assets}x{n_assets}")
            for j, row in enumerate(regime.correlation_matrix):
                if len(row) != n_assets:
                    raise ValueError(f"Regime {i} correlation_matrix row {j} length mismatch")
            # Validate correlation matrix symmetry and diagonal
            corr = np.array(regime.correlation_matrix)
            if not np.allclose(corr, corr.T, atol=1e-8):
                raise ValueError(f"Regime {i} correlation matrix must be symmetric")
            if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
                raise ValueError(f"Regime {i} correlation matrix diagonal must be 1.0")
        return self


class GuaranteedIncomeStream(BaseModel):
    """A recurring income stream (Social Security, pension, annuity)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Stream name (e.g. 'Social Security', 'Pension')")
    monthly_amount: float = Field(gt=0, description="Monthly benefit in today's dollars")
    start_age: float = Field(ge=18, le=120, description="Age when payments begin")
    cola_rate: float = Field(
        default=0.0,
        ge=0,
        le=0.10,
        description="Annual cost-of-living adjustment rate (0 = no COLA)",
    )
    end_age: float | None = Field(
        default=None,
        ge=18,
        le=120,
        description="Age when payments stop (None = lifetime)",
    )


class DiscreteEvent(BaseModel):
    """A one-time financial event at a specific age."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=18, le=120, description="Age when event occurs")
    amount: float = Field(description="Dollar amount (positive=inflow, negative=outflow)")
    description: str = Field(default="", description="Description of the event")


class AccountConfig(BaseModel):
    """A single investment account."""

    model_config = ConfigDict(extra="forbid")

    account_type: Literal["taxable", "traditional", "roth"] = "taxable"
    balance: float = Field(ge=0, description="Current balance in dollars")
    annual_contribution: float = Field(
        default=0.0, ge=0, description="Annual contribution in dollars"
    )


class PlanConfig(BaseModel):
    """Core financial plan parameters."""

    model_config = ConfigDict(extra="forbid")

    current_age: int = Field(ge=18, le=100)
    retirement_age: int = Field(ge=18, le=100)
    end_age: int = Field(ge=18, le=120)
    accounts: list[AccountConfig] = Field(min_length=1)
    monthly_income: float = Field(default=0.0, ge=0)
    monthly_spending: float = Field(ge=0)
    income_growth_rate: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Annual real income growth rate (e.g. 0.02 for 2%)",
    )
    discrete_events: list[DiscreteEvent] = Field(
        default_factory=list,
        description="One-time financial events (e.g. home purchase, inheritance)",
    )
    income_end_age: int | None = Field(
        default=None, ge=18, le=120, description="Age when earned income stops"
    )
    guaranteed_income: list[GuaranteedIncomeStream] = Field(
        default_factory=list,
        description="Recurring income streams (Social Security, pensions, annuities)",
    )

    @model_validator(mode="after")
    def _validate_ages(self) -> PlanConfig:
        if self.end_age <= self.retirement_age:
            raise ValueError("end_age must be greater than retirement_age")
        if self.income_end_age is None:
            self.income_end_age = self.retirement_age
        if self.income_end_age > self.end_age:
            raise ValueError("income_end_age must not exceed end_age")
        return self


class AssetClass(BaseModel):
    """An asset class with a target allocation weight."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = Field(ge=0, le=1)


class GlidePath(BaseModel):
    """Age-based target allocation that shifts over time."""

    model_config = ConfigDict(extra="forbid")

    start_age: int = Field(ge=18, le=120)
    start_weights: list[float] = Field(min_length=1)
    end_age: int = Field(ge=18, le=120)
    end_weights: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_glide_path(self) -> GlidePath:
        if self.end_age <= self.start_age:
            raise ValueError("end_age must be greater than start_age")
        if len(self.start_weights) != len(self.end_weights):
            raise ValueError("start_weights and end_weights must have same length")
        if abs(sum(self.start_weights) - 1.0) > 1e-6:
            raise ValueError(f"start_weights must sum to 1.0, got {sum(self.start_weights)}")
        if abs(sum(self.end_weights) - 1.0) > 1e-6:
            raise ValueError(f"end_weights must sum to 1.0, got {sum(self.end_weights)}")
        return self


class AssetAllocation(BaseModel):
    """The asset allocation and glide path for one specific account."""

    model_config = ConfigDict(extra="forbid")

    assets: list[AssetClass] = Field(min_length=1)
    glide_path: GlidePath | None = Field(
        default=None,
        description="Age-based glide path; when set, overrides static asset weights over time",
    )


class MarketAssumptions(BaseModel):
    """Market return and inflation assumptions."""

    model_config = ConfigDict(extra="forbid")

    asset_allocations: list[AssetAllocation] = Field(
        min_length=1,
        description=(
            "One asset allocation per account, or a single allocation "
            "to broadcast to all accounts."
        ),
    )
    expected_annual_returns: list[float]
    annual_volatilities: list[float]
    correlation_matrix: list[list[float]]
    inflation_mean: float = Field(default=0.03, description="Long-run annual inflation rate")
    inflation_vol: float = Field(default=0.01, ge=0, description="Inflation annual volatility")
    return_model: Literal["mvn", "student_t", "bootstrap", "regime_switching"] = Field(
        default="mvn",
        description="Return model type",
    )
    degrees_of_freedom: float | None = Field(
        default=None,
        gt=2,
        description="Degrees of freedom for student-t returns",
    )
    historical_returns: list[list[float]] | None = Field(
        default=None,
        description="Historical monthly returns matrix (n_months x n_assets) for bootstrap",
    )
    bootstrap_block_size: int = Field(
        default=12,
        ge=1,
        description="Block size for block bootstrap (months)",
    )
    regime_switching: RegimeSwitchingConfig | None = Field(
        default=None,
        description="Regime-switching model configuration",
    )
    expense_ratio: float = Field(
        default=0.0,
        ge=0,
        le=0.05,
        description="Annual fund expense ratio (e.g. 0.001 for 10bps)",
    )
    aum_fee: float = Field(
        default=0.0,
        ge=0,
        le=0.05,
        description="Annual AUM/advisory platform fee",
    )
    advisory_fee: float = Field(
        default=0.0,
        ge=0,
        le=0.05,
        description="Annual financial advisor fee",
    )

    @model_validator(mode="after")
    def _validate_market(self) -> MarketAssumptions:
        n = len(self.expected_annual_returns)
        for i, alloc in enumerate(self.asset_allocations):
            if len(alloc.assets) != n:
                raise ValueError(
                    f"asset_allocations[{i}] length ({len(alloc.assets)}) "
                    f"must match expected_annual_returns length ({n})"
                )
            total_weight = sum(a.weight for a in alloc.assets)
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(
                    f"Asset weights in allocation {i} must sum to 1.0, got {total_weight}"
                )

        if len(self.annual_volatilities) != n:
            raise ValueError(
                f"annual_volatilities length ({len(self.annual_volatilities)}) "
                f"must match expected_annual_returns length ({n})"
            )
        if len(self.correlation_matrix) != n:
            raise ValueError(
                f"correlation_matrix must be {n}x{n}, got {len(self.correlation_matrix)} rows"
            )
        for i, row in enumerate(self.correlation_matrix):
            if len(row) != n:
                raise ValueError(f"correlation_matrix row {i} has length {len(row)}, expected {n}")

        # Validate correlation matrix is symmetric and has 1s on diagonal
        corr = np.array(self.correlation_matrix)
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValueError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("Correlation matrix diagonal must be 1.0")

        return self


class StressScenario(BaseModel):
    """A deterministic stress scenario overlay."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Scenario name")
    scenario_type: Literal["crash", "lost_decade", "high_inflation", "sequence_risk"] = Field(
        description="Type of stress scenario",
    )
    start_age: float = Field(ge=18, le=120, description="Age when scenario begins")
    duration_months: int = Field(ge=1, le=360, description="Duration in months")
    severity: float = Field(
        default=1.0,
        gt=0,
        description="Severity multiplier (1.0 = default scenario intensity)",
    )


class SimulationConfig(BaseModel):
    """Simulation execution parameters."""

    model_config = ConfigDict(extra="forbid")

    n_paths: int = Field(default=5000, ge=1, le=1_000_000)
    seed: int = Field(default=42, ge=0)
    store_paths: bool = Field(default=False, description="Store full path data (memory-heavy)")
    antithetic: bool = Field(
        default=False,
        description="Use antithetic variates for variance reduction",
    )
    preset: Literal["fast", "balanced", "deep"] | None = Field(
        default=None,
        description="Simulation quality preset (overrides n_paths and antithetic)",
    )
    stress_scenarios: list[StressScenario] = Field(
        default_factory=list,
        description="Deterministic stress scenarios to apply",
    )

    @model_validator(mode="after")
    def _apply_preset(self) -> SimulationConfig:
        if self.preset is not None:
            preset_configs = {
                "fast": (1000, False),
                "balanced": (5000, False),
                "deep": (20000, True),
            }
            n_paths, antithetic = preset_configs[self.preset]
            self.n_paths = n_paths
            self.antithetic = antithetic
        return self


class GuardrailsConfig(BaseModel):
    """Guyton-Klinger guardrails spending policy parameters."""

    model_config = ConfigDict(extra="forbid")

    initial_withdrawal_rate: float = Field(default=0.05, gt=0, le=0.2)
    upper_threshold: float = Field(default=0.20, gt=0, le=1)
    lower_threshold: float = Field(default=0.20, gt=0, le=1)
    raise_pct: float = Field(default=0.10, gt=0, le=1)
    cut_pct: float = Field(default=0.10, gt=0, le=1)


class VPWConfig(BaseModel):
    """Variable Percentage Withdrawal spending policy parameters."""

    model_config = ConfigDict(extra="forbid")

    min_rate: float = Field(default=0.03, ge=0, le=1)
    max_rate: float = Field(default=0.15, ge=0, le=1)


class FloorCeilingConfig(BaseModel):
    """Floor-and-ceiling spending policy parameters."""

    model_config = ConfigDict(extra="forbid")

    withdrawal_rate: float = Field(default=0.04, gt=0, le=0.2)
    floor: float = Field(
        default=3000.0,
        ge=0,
        description="Monthly spending floor in today's dollars",
    )
    ceiling: float = Field(
        default=10000.0,
        ge=0,
        description="Monthly spending ceiling in today's dollars",
    )


class SpendingPolicyConfig(BaseModel):
    """Spending policy configuration."""

    model_config = ConfigDict(extra="forbid")

    policy_type: Literal[
        "constant_real",
        "percent_of_portfolio",
        "guardrails",
        "vpw",
        "floor_ceiling",
    ] = "constant_real"
    withdrawal_rate: float = Field(
        default=0.04,
        gt=0,
        le=1,
        description="Annual withdrawal rate for percent_of_portfolio policy",
    )
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    vpw: VPWConfig = Field(default_factory=VPWConfig)
    floor_ceiling: FloorCeilingConfig = Field(default_factory=FloorCeilingConfig)


class RothConversionConfig(BaseModel):
    """Roth conversion strategy configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable Roth conversions")
    strategy: Literal["fixed_amount", "fill_bracket"] = Field(
        default="fixed_amount",
        description="Conversion strategy: fixed annual amount or fill to bracket top",
    )
    annual_amount: float = Field(
        default=0.0,
        ge=0,
        description="Annual conversion amount (fixed_amount strategy)",
    )
    fill_to_bracket_top: float = Field(
        default=0.22,
        ge=0.10,
        le=0.37,
        description="Target marginal rate for fill_bracket strategy",
    )
    start_age: float = Field(
        default=55,
        ge=18,
        le=120,
        description="Age to begin Roth conversions",
    )
    end_age: float = Field(
        default=72,
        ge=18,
        le=120,
        description="Age to stop Roth conversions",
    )

    @model_validator(mode="after")
    def _validate_ages(self) -> RothConversionConfig:
        if self.end_age <= self.start_age:
            raise ValueError("end_age must be greater than start_age")
        return self


class PolicyBundle(BaseModel):
    """Bundle of all policy configurations."""

    model_config = ConfigDict(extra="forbid")

    spending: SpendingPolicyConfig = Field(default_factory=SpendingPolicyConfig)
    rebalancing_strategy: Literal["calendar", "threshold"] = Field(
        default="calendar",
        description="Rebalancing trigger: calendar schedule or drift threshold",
    )
    rebalancing_months: list[int] = Field(
        default=[1, 7],
        description="Months (1-12) when rebalancing occurs (calendar strategy)",
    )
    rebalancing_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.50,
        description="Max drift before rebalancing (threshold strategy, e.g. 0.05 = 5%)",
    )
    withdrawal_order: list[Literal["taxable", "traditional", "roth"]] = Field(
        default=["taxable", "traditional", "roth"],
        description="Account withdrawal priority order",
    )
    tax_model: Literal["flat", "us_federal"] = Field(
        default="flat",
        description="Tax model to use",
    )
    tax_rate: float = Field(
        default=0.22,
        ge=0,
        le=1,
        description="Flat effective tax rate (used when tax_model='flat')",
    )
    filing_status: Literal["single", "married_jointly"] = Field(
        default="single",
        description="Tax filing status (used when tax_model='us_federal')",
    )
    state_tax_rate: float = Field(
        default=0.0,
        ge=0,
        le=0.15,
        description="Flat state income tax rate on ordinary income + LTCG",
    )
    include_niit: bool = Field(
        default=False,
        description="Include 3.8% Net Investment Income Tax surtax",
    )
    roth_conversion: RothConversionConfig = Field(
        default_factory=RothConversionConfig,
        description="Roth conversion strategy configuration",
    )
