"""Default configuration values for monteplan."""

from __future__ import annotations

from monteplan.config.schema import (
    AccountConfig,
    AssetAllocation,
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)

# --- 6-Asset Global Diversification Constants ---

STOCK_ASSET_NAMES: list[str] = ["US Stocks", "Ex-US Dev Stocks", "EM Stocks"]
BOND_ASSET_NAMES: list[str] = ["US Bonds", "Ex-US Dev Bonds", "EM Bonds"]
ALL_ASSET_NAMES: list[str] = STOCK_ASSET_NAMES + BOND_ASSET_NAMES

# Default regional split: 60% US / 30% Ex-US Dev / 10% EM
DEFAULT_REGIONAL_SPLIT: list[float] = [0.60, 0.30, 0.10]

# Nominal expected returns (real + 3% inflation baseline)
_STOCK_RETURNS_NOMINAL: list[float] = [0.08, 0.075, 0.085]
_BOND_RETURNS_AGGREGATE: list[float] = [0.05, 0.045, 0.06]
_BOND_RETURNS_TREASURIES: list[float] = [0.045, 0.045, 0.06]

_STOCK_VOLS: list[float] = [0.16, 0.18, 0.23]
_BOND_VOLS_AGGREGATE: list[float] = [0.07, 0.09, 0.12]
_BOND_VOLS_TREASURIES: list[float] = [0.06, 0.09, 0.12]

# Academic Consensus correlation matrix (6x6)
#               US_Stk  ExUS_Stk  EM_Stk  US_Bnd  ExUS_Bnd  EM_Bnd
CORRELATION_CONSENSUS: list[list[float]] = [
    [1.00, 0.80, 0.65, 0.00, -0.05, 0.10],  # US Stocks
    [0.80, 1.00, 0.75, -0.05, 0.00, 0.15],  # Ex-US Dev Stocks
    [0.65, 0.75, 1.00, 0.05, 0.05, 0.30],  # EM Stocks
    [0.00, -0.05, 0.05, 1.00, 0.65, 0.40],  # US Bonds
    [-0.05, 0.00, 0.05, 0.65, 1.00, 0.50],  # Ex-US Dev Bonds
    [0.10, 0.15, 0.30, 0.40, 0.50, 1.00],  # EM Bonds
]

# Crisis-Aware correlation matrix (higher cross-correlations)
CORRELATION_CRISIS_AWARE: list[list[float]] = [
    [1.00, 0.90, 0.80, 0.10, 0.05, 0.20],  # US Stocks
    [0.90, 1.00, 0.85, 0.05, 0.10, 0.25],  # Ex-US Dev Stocks
    [0.80, 0.85, 1.00, 0.15, 0.10, 0.40],  # EM Stocks
    [0.10, 0.05, 0.15, 1.00, 0.75, 0.50],  # US Bonds
    [0.05, 0.10, 0.10, 0.75, 1.00, 0.60],  # Ex-US Dev Bonds
    [0.20, 0.25, 0.40, 0.50, 0.60, 1.00],  # EM Bonds
]


def build_global_weights(
    stock_pct: float = 0.60,
    stock_regional: list[float] | None = None,
    bond_regional: list[float] | None = None,
) -> list[float]:
    """Build 6-asset weight vector from stock/bond split and regional ratios.

    Args:
        stock_pct: Fraction allocated to equities (0-1). Bonds get the rest.
        stock_regional: Regional split within stocks [US, Ex-US Dev, EM].
            Must sum to 1. Defaults to [0.60, 0.30, 0.10].
        bond_regional: Regional split within bonds [US, Ex-US Dev, EM].
            Must sum to 1. Defaults to stock_regional if None.

    Returns:
        List of 6 weights: [US_Stk, ExUS_Stk, EM_Stk, US_Bnd, ExUS_Bnd, EM_Bnd].
    """
    if stock_regional is None:
        stock_regional = list(DEFAULT_REGIONAL_SPLIT)
    if bond_regional is None:
        bond_regional = list(stock_regional)

    bond_pct = 1.0 - stock_pct
    return [s * stock_pct for s in stock_regional] + [b * bond_pct for b in bond_regional]


def default_plan() -> PlanConfig:
    """Default plan: 30-year-old, retiring at 65, horizon to 95."""
    return PlanConfig(
        current_age=30,
        retirement_age=65,
        end_age=95,
        accounts=[
            AccountConfig(account_type="taxable", balance=50_000, annual_contribution=6_000),
            AccountConfig(account_type="traditional", balance=100_000, annual_contribution=20_000),
            AccountConfig(account_type="roth", balance=30_000, annual_contribution=7_000),
        ],
        monthly_income=8_000,
        monthly_spending=5_000,
    )


def default_market(bond_type: str = "aggregate") -> MarketAssumptions:
    """Default 6-asset global market: US/Ex-US Dev/EM for stocks and bonds.

    Args:
        bond_type: "aggregate" (default) or "treasuries". Controls US bond
            return and volatility assumptions.

    Returns:
        MarketAssumptions with 6 assets and academic-consensus correlations.
    """
    weights = build_global_weights()
    assets = [
        AssetClass(name=name, weight=w) for name, w in zip(ALL_ASSET_NAMES, weights, strict=True)
    ]

    if bond_type == "treasuries":
        bond_returns = list(_BOND_RETURNS_TREASURIES)
        bond_vols = list(_BOND_VOLS_TREASURIES)
    else:
        bond_returns = list(_BOND_RETURNS_AGGREGATE)
        bond_vols = list(_BOND_VOLS_AGGREGATE)

    # We supply 1 asset allocation here; the engine will automatically broadcast this
    # to all N accounts in the plan when simulate() is called.
    asset_alloc = AssetAllocation(assets=assets)

    return MarketAssumptions(
        asset_allocations=[asset_alloc],
        expected_annual_returns=list(_STOCK_RETURNS_NOMINAL) + bond_returns,
        annual_volatilities=list(_STOCK_VOLS) + bond_vols,
        correlation_matrix=[list(row) for row in CORRELATION_CONSENSUS],
        inflation_mean=0.03,
        inflation_vol=0.01,
    )


def global_market(bond_type: str = "aggregate") -> MarketAssumptions:
    """Alias for default_market() — 6-asset global diversified market."""
    return default_market(bond_type=bond_type)


def us_only_market(bond_type: str = "aggregate") -> MarketAssumptions:
    """2-asset US-only market: US Stocks + US Bonds.

    Return assumptions aligned with the global defaults (Damodaran-based).

    Args:
        bond_type: "aggregate" (default, 5% nominal) or "treasuries" (4.5% nominal).
    """
    if bond_type == "treasuries":
        bond_return = 0.045
        bond_vol = 0.06
    else:
        bond_return = 0.05
        bond_vol = 0.07

    # We supply 1 asset allocation here; the engine will automatically broadcast this
    # to all N accounts in the plan when simulate() is called.
    asset_alloc = AssetAllocation(
        assets=[
            AssetClass(name="US Stocks", weight=0.7),
            AssetClass(name="US Bonds", weight=0.3),
        ]
    )

    return MarketAssumptions(
        asset_allocations=[asset_alloc],
        expected_annual_returns=[0.08, bond_return],
        annual_volatilities=[0.16, bond_vol],
        correlation_matrix=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        inflation_mean=0.03,
        inflation_vol=0.01,
    )


def default_sim_config() -> SimulationConfig:
    """Default simulation config: 5000 paths, seed 42."""
    return SimulationConfig(n_paths=5000, seed=42)


def default_policies() -> PolicyBundle:
    """Default policies: constant real spending, semi-annual rebalancing."""
    return PolicyBundle()


# --- Quick Start Templates ---


def fire_plan() -> PlanConfig:
    """FIRE template: aggressive saver, early retirement at 45."""
    return PlanConfig(
        current_age=30,
        retirement_age=45,
        end_age=90,
        accounts=[
            AccountConfig(account_type="taxable", balance=200_000, annual_contribution=30_000),
            AccountConfig(account_type="traditional", balance=100_000, annual_contribution=23_000),
            AccountConfig(account_type="roth", balance=50_000, annual_contribution=7_000),
        ],
        monthly_income=12_000,
        monthly_spending=3_500,
        income_growth_rate=0.03,
    )


def coast_fire_plan() -> PlanConfig:
    """Coast FIRE template: stop contributing, let investments grow."""
    return PlanConfig(
        current_age=35,
        retirement_age=60,
        end_age=95,
        accounts=[
            AccountConfig(account_type="taxable", balance=100_000, annual_contribution=0),
            AccountConfig(account_type="traditional", balance=300_000, annual_contribution=0),
            AccountConfig(account_type="roth", balance=80_000, annual_contribution=0),
        ],
        monthly_income=7_000,
        monthly_spending=4_000,
    )


def conservative_retiree_plan() -> PlanConfig:
    """Conservative retiree template: near retirement with Social Security."""
    from monteplan.config.schema import GuaranteedIncomeStream

    return PlanConfig(
        current_age=60,
        retirement_age=65,
        end_age=95,
        accounts=[
            AccountConfig(account_type="taxable", balance=200_000, annual_contribution=5_000),
            AccountConfig(account_type="traditional", balance=500_000, annual_contribution=20_000),
        ],
        monthly_income=8_000,
        monthly_spending=5_000,
        guaranteed_income=[
            GuaranteedIncomeStream(
                name="Social Security",
                monthly_amount=2_500,
                start_age=67,
                cola_rate=0.02,
            ),
        ],
    )
