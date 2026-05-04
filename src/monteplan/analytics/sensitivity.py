"""One-at-a-time (OAT) sensitivity analysis for simulation parameters."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from monteplan.config.schema import (
    AssetClass,
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.engine import simulate

# Type aliases for getter/setter callables
_Getter = Callable[..., float]
_Setter = Callable[..., Any]


@dataclass(frozen=True)
class SensitivityResult:
    """Result of perturbing a single parameter."""

    parameter_name: str
    base_value: float
    low_value: float
    high_value: float
    base_success: float
    low_success: float
    high_success: float
    impact: float  # high_success - low_success


@dataclass
class SensitivityReport:
    """Full report of one-at-a-time sensitivity analysis."""

    results: list[SensitivityResult] = field(default_factory=list)
    base_success_probability: float = 0.0


@dataclass(frozen=True)
class HeatmapResult:
    """Result of a 2D sensitivity grid analysis."""

    x_param_name: str
    y_param_name: str
    x_values: list[float]
    y_values: list[float]
    success_grid: list[list[float]]  # [y_idx][x_idx]
    base_x_value: float
    base_y_value: float
    base_success: float


@dataclass
class _ParamSpec:
    """Internal spec for a perturbable parameter."""

    getter: _Getter
    setter: _Setter
    is_additive: bool = False
    additive_delta: float = 0.0
    target: str = "market"  # "market", "plan", or "policies"


def _make_market_return_setter(idx: int) -> _Setter:
    def setter(m: MarketAssumptions, val: float) -> MarketAssumptions:
        new_returns = list(m.expected_annual_returns)
        new_returns[idx] = val
        return m.model_copy(update={"expected_annual_returns": new_returns})

    return setter


def _make_market_vol_setter(idx: int) -> _Setter:
    def setter(m: MarketAssumptions, val: float) -> MarketAssumptions:
        new_vols = list(m.annual_volatilities)
        new_vols[idx] = val
        return m.model_copy(update={"annual_volatilities": new_vols})

    return setter


def _make_contribution_setter(idx: int) -> _Setter:
    def setter(p: PlanConfig, val: float) -> PlanConfig:
        new_accounts = list(p.accounts)
        new_accounts[idx] = new_accounts[idx].model_copy(
            update={"annual_contribution": max(0.0, val)}
        )
        return p.model_copy(update={"accounts": new_accounts})

    return setter


def _equity_allocation_setter(m: MarketAssumptions, val: float) -> MarketAssumptions:
    """Set total equity allocation, scaling stock and bond groups proportionally.

    Identifies stock vs bond assets by name substring ("Stock" / "Bond").
    Preserves regional ratios within each group. For 2-asset portfolios,
    behaves identically to the old _stock_allocation_setter.
    """
    clamped = max(0.0, min(1.0, val))
    bond_total = 1.0 - clamped

    stock_indices = [i for i, a in enumerate(m.asset_allocations[0].assets) if "Stock" in a.name]
    bond_indices = [i for i, a in enumerate(m.asset_allocations[0].assets) if "Bond" in a.name]

    # Current totals for proportional scaling
    current_stock_total = sum(m.asset_allocations[0].assets[i].weight for i in stock_indices)
    current_bond_total = sum(m.asset_allocations[0].assets[i].weight for i in bond_indices)

    new_assets = list(m.asset_allocations[0].assets)
    for i in stock_indices:
        if current_stock_total > 0:
            ratio = m.asset_allocations[0].assets[i].weight / current_stock_total
        else:
            ratio = 1.0 / len(stock_indices) if stock_indices else 0.0
        new_assets[i] = AssetClass(name=new_assets[i].name, weight=clamped * ratio)

    for i in bond_indices:
        if current_bond_total > 0:
            ratio = m.asset_allocations[0].assets[i].weight / current_bond_total
        else:
            ratio = 1.0 / len(bond_indices) if bond_indices else 0.0
        new_assets[i] = AssetClass(name=new_assets[i].name, weight=bond_total * ratio)

    new_allocations = list(m.asset_allocations)
    new_allocations[0] = new_allocations[0].model_copy(update={"assets": new_assets})
    return m.model_copy(update={"asset_allocations": new_allocations})


def _build_param_registry(
    plan: PlanConfig,
    market: MarketAssumptions,
    policies: PolicyBundle | None = None,
) -> dict[str, _ParamSpec]:
    """Build the full registry of perturbable parameters."""
    all_params: dict[str, _ParamSpec] = {}
    n_assets = len(market.asset_allocations[0].assets)

    for i in range(n_assets):
        asset_name = market.asset_allocations[0].assets[i].name
        all_params[f"{asset_name} Return"] = _ParamSpec(
            getter=lambda m, idx=i: m.expected_annual_returns[idx],
            setter=_make_market_return_setter(i),
            is_additive=False,
        )
        all_params[f"{asset_name} Volatility"] = _ParamSpec(
            getter=lambda m, idx=i: m.annual_volatilities[idx],
            setter=_make_market_vol_setter(i),
            is_additive=False,
        )

    all_params["Inflation Rate"] = _ParamSpec(
        getter=lambda m: m.inflation_mean,
        setter=lambda m, v: m.model_copy(update={"inflation_mean": v}),
        is_additive=False,
    )

    all_params["Monthly Spending"] = _ParamSpec(
        getter=lambda p: p.monthly_spending,
        setter=lambda p, v: p.model_copy(update={"monthly_spending": v}),
        is_additive=False,
        target="plan",
    )

    all_params["Retirement Age"] = _ParamSpec(
        getter=lambda p: float(p.retirement_age),
        setter=lambda p, v: p.model_copy(update={"retirement_age": int(round(v))}),
        is_additive=True,
        additive_delta=2.0,
        target="plan",
    )

    for i, acct in enumerate(plan.accounts):
        if acct.annual_contribution > 0:
            all_params[f"{acct.account_type.title()} Contribution"] = _ParamSpec(
                getter=lambda p, idx=i: p.accounts[idx].annual_contribution,
                setter=_make_contribution_setter(i),
                is_additive=False,
                target="plan",
            )

    # Equity allocation (total stock weight)
    if n_assets >= 2:
        all_params["Equity Allocation"] = _ParamSpec(
            getter=lambda m: sum(
                a.weight for a in m.asset_allocations[0].assets if "Stock" in a.name
            ),
            setter=_equity_allocation_setter,
            is_additive=False,
        )

    # State tax rate (only register when non-zero)
    if policies is not None and policies.state_tax_rate > 0:
        all_params["State Tax Rate"] = _ParamSpec(
            getter=lambda p: p.state_tax_rate,
            setter=lambda p, v: p.model_copy(update={"state_tax_rate": max(0.0, min(0.15, v))}),
            is_additive=True,
            additive_delta=0.02,
            target="policies",
        )

    return all_params


def _run_one(
    plan: PlanConfig,
    market: MarketAssumptions,
    policies: PolicyBundle,
    sim_config: SimulationConfig,
) -> float:
    """Run a single simulation and return success probability.

    Top-level function so it is picklable for ProcessPoolExecutor.
    """
    try:
        result = simulate(plan, market, policies, sim_config)
        return result.success_probability
    except Exception:
        return -1.0  # sentinel for failure


def run_sensitivity(
    plan: PlanConfig,
    market: MarketAssumptions,
    policies: PolicyBundle,
    sim_config: SimulationConfig,
    perturbation_pct: float = 0.10,
    parameters: list[str] | None = None,
    max_workers: int | None = None,
) -> SensitivityReport:
    """Run OAT sensitivity analysis.

    Perturbs each parameter by ±perturbation_pct (or additive for ages)
    and measures the impact on success probability.

    Args:
        plan: Financial plan configuration.
        market: Market assumptions.
        policies: Policy bundle.
        sim_config: Simulation config (n_paths capped at 2000 for speed).
        perturbation_pct: Fractional perturbation (default 10%).
        parameters: Optional list of parameter names to vary. If None,
            uses all default parameters.
        max_workers: Maximum number of parallel workers. ``None`` uses
            all available cores; ``1`` forces sequential execution.

    Returns:
        SensitivityReport with per-parameter results.
    """
    # Cap paths for speed
    capped_paths = min(sim_config.n_paths, 2000)
    base_sim = sim_config.model_copy(update={"n_paths": capped_paths, "preset": None})

    # Base run
    base_result = simulate(plan, market, policies, base_sim)
    base_success = base_result.success_probability

    # Define all perturbable parameters
    all_params = _build_param_registry(plan, market, policies)

    # Filter to requested parameters
    if parameters is not None:
        param_set = {k: v for k, v in all_params.items() if k in parameters}
    else:
        param_set = all_params

    # Build all perturbation jobs upfront (picklable tuples)
    # Each job: (param_name, direction, plan, market, policies, sim_config)
    jobs: list[tuple[str, str, PlanConfig, MarketAssumptions, PolicyBundle, SimulationConfig]] = []
    param_vals: dict[str, tuple[float, float, float]] = {}  # name -> (base, low, high)

    for param_name, spec in param_set.items():
        if spec.target == "plan":
            target_obj: PlanConfig | MarketAssumptions | PolicyBundle = plan
        elif spec.target == "policies":
            target_obj = policies
        else:
            target_obj = market
        base_val: float = spec.getter(target_obj)

        if spec.is_additive:
            delta = spec.additive_delta
            low_val = base_val - delta
            high_val = base_val + delta
        else:
            low_val = base_val * (1.0 - perturbation_pct)
            high_val = base_val * (1.0 + perturbation_pct)

        param_vals[param_name] = (base_val, low_val, high_val)

        # Build low perturbation config
        if spec.target == "plan":
            try:
                low_plan: PlanConfig = spec.setter(plan, low_val)
                jobs.append((param_name, "low", low_plan, market, policies, base_sim))
            except Exception:
                jobs.append((param_name, "low_fail", plan, market, policies, base_sim))
        elif spec.target == "policies":
            try:
                low_policies: PolicyBundle = spec.setter(policies, low_val)
                jobs.append((param_name, "low", plan, market, low_policies, base_sim))
            except Exception:
                jobs.append((param_name, "low_fail", plan, market, policies, base_sim))
        else:
            try:
                low_market: MarketAssumptions = spec.setter(market, low_val)
                jobs.append((param_name, "low", plan, low_market, policies, base_sim))
            except Exception:
                jobs.append((param_name, "low_fail", plan, market, policies, base_sim))

        # Build high perturbation config
        if spec.target == "plan":
            try:
                high_plan: PlanConfig = spec.setter(plan, high_val)
                jobs.append((param_name, "high", high_plan, market, policies, base_sim))
            except Exception:
                jobs.append((param_name, "high_fail", plan, market, policies, base_sim))
        elif spec.target == "policies":
            try:
                high_policies: PolicyBundle = spec.setter(policies, high_val)
                jobs.append((param_name, "high", plan, market, high_policies, base_sim))
            except Exception:
                jobs.append((param_name, "high_fail", plan, market, policies, base_sim))
        else:
            try:
                high_market: MarketAssumptions = spec.setter(market, high_val)
                jobs.append((param_name, "high", plan, high_market, policies, base_sim))
            except Exception:
                jobs.append((param_name, "high_fail", plan, market, policies, base_sim))

    # Run all perturbation simulations
    results_map: dict[tuple[str, str], float] = {}

    if max_workers == 1:
        # Sequential execution
        for param_name, direction, j_plan, j_market, j_policies, j_sim in jobs:
            if direction.endswith("_fail"):
                results_map[(param_name, direction.replace("_fail", ""))] = base_success
            else:
                results_map[(param_name, direction)] = _run_one(
                    j_plan, j_market, j_policies, j_sim
                )
    else:
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_key: dict[concurrent.futures.Future[float], tuple[str, str]] = {}
            for param_name, direction, j_plan, j_market, j_policies, j_sim in jobs:
                if direction.endswith("_fail"):
                    results_map[(param_name, direction.replace("_fail", ""))] = base_success
                else:
                    future = executor.submit(_run_one, j_plan, j_market, j_policies, j_sim)
                    future_to_key[future] = (param_name, direction)

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                success = future.result()
                if success < 0:
                    # Sentinel: simulation failed, use base success
                    results_map[key] = base_success
                else:
                    results_map[key] = success

    # Assemble report in original parameter order
    report = SensitivityReport(base_success_probability=base_success)

    for param_name in param_set:
        base_val, low_val, high_val = param_vals[param_name]
        low_success = results_map.get((param_name, "low"), base_success)
        high_success = results_map.get((param_name, "high"), base_success)

        report.results.append(
            SensitivityResult(
                parameter_name=param_name,
                base_value=base_val,
                low_value=low_val,
                high_value=high_val,
                base_success=base_success,
                low_success=low_success,
                high_success=high_success,
                impact=high_success - low_success,
            )
        )

    return report


def run_2d_sensitivity(
    plan: PlanConfig,
    market: MarketAssumptions,
    policies: PolicyBundle,
    sim_config: SimulationConfig,
    x_param: str,
    y_param: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_steps: int = 12,
    y_steps: int = 12,
    max_workers: int | None = None,
) -> HeatmapResult:
    """Run a 2D grid sensitivity analysis.

    Varies two parameters simultaneously and measures success probability
    across an x_steps x y_steps grid.

    Args:
        plan: Financial plan configuration.
        market: Market assumptions.
        policies: Policy bundle.
        sim_config: Simulation config (n_paths capped at 1000 for speed).
        x_param: Name of the x-axis parameter.
        y_param: Name of the y-axis parameter.
        x_range: (min, max) range for x parameter.
        y_range: (min, max) range for y parameter.
        x_steps: Number of grid points along x-axis.
        y_steps: Number of grid points along y-axis.
        max_workers: Maximum number of parallel workers.

    Returns:
        HeatmapResult with the success probability grid.
    """
    # Cap paths for speed
    capped_paths = min(sim_config.n_paths, 1000)
    base_sim = sim_config.model_copy(update={"n_paths": capped_paths, "preset": None})

    all_params = _build_param_registry(plan, market, policies)

    if x_param not in all_params:
        raise ValueError(f"Unknown parameter: {x_param}")
    if y_param not in all_params:
        raise ValueError(f"Unknown parameter: {y_param}")

    x_spec = all_params[x_param]
    y_spec = all_params[y_param]

    x_values = np.linspace(x_range[0], x_range[1], x_steps).tolist()
    y_values = np.linspace(y_range[0], y_range[1], y_steps).tolist()

    # Get base values
    def _resolve_target(
        spec: _ParamSpec,
    ) -> PlanConfig | MarketAssumptions | PolicyBundle:
        if spec.target == "plan":
            return plan
        if spec.target == "policies":
            return policies
        return market

    base_x = x_spec.getter(_resolve_target(x_spec))
    base_y = y_spec.getter(_resolve_target(y_spec))

    # Base run
    base_result = simulate(plan, market, policies, base_sim)
    base_success = base_result.success_probability * 100

    # Build all grid jobs
    jobs: list[tuple[int, int, PlanConfig, MarketAssumptions, PolicyBundle, SimulationConfig]] = []
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            cur_plan = plan
            cur_market = market
            cur_policies = policies
            # Apply x parameter
            if x_spec.target == "plan":
                cur_plan = x_spec.setter(cur_plan, xv)
            elif x_spec.target == "policies":
                cur_policies = x_spec.setter(cur_policies, xv)
            else:
                cur_market = x_spec.setter(cur_market, xv)
            # Apply y parameter
            if y_spec.target == "plan":
                cur_plan = y_spec.setter(cur_plan, yv)
            elif y_spec.target == "policies":
                cur_policies = y_spec.setter(cur_policies, yv)
            else:
                cur_market = y_spec.setter(cur_market, yv)
            jobs.append((yi, xi, cur_plan, cur_market, cur_policies, base_sim))

    # Run all jobs
    grid: list[list[float]] = [[0.0] * x_steps for _ in range(y_steps)]

    if max_workers == 1:
        for yi, xi, j_plan, j_market, j_policies, j_sim in jobs:
            success = _run_one(j_plan, j_market, j_policies, j_sim)
            grid[yi][xi] = max(success, 0.0) * 100
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx: dict[concurrent.futures.Future[float], tuple[int, int]] = {}
            for yi, xi, j_plan, j_market, j_policies, j_sim in jobs:
                future = executor.submit(_run_one, j_plan, j_market, j_policies, j_sim)
                future_to_idx[future] = (yi, xi)

            for future in concurrent.futures.as_completed(future_to_idx):
                yi, xi = future_to_idx[future]
                success = future.result()
                grid[yi][xi] = max(success, 0.0) * 100

    return HeatmapResult(
        x_param_name=x_param,
        y_param_name=y_param,
        x_values=x_values,
        y_values=y_values,
        success_grid=grid,
        base_x_value=base_x,
        base_y_value=base_y,
        base_success=base_success,
    )
