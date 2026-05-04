"""Core simulation engine — the heart of monteplan."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monteplan import __version__
from monteplan.config.schema import (
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)
from monteplan.core.rng import make_rng
from monteplan.core.state import SimulationState
from monteplan.core.timeline import Timeline
from monteplan.io.serialize import compute_config_hash
from monteplan.models.inflation import OUInflationModel, RegimeSwitchingInflationModel
from monteplan.models.returns.bootstrap import HistoricalBootstrapReturns
from monteplan.models.returns.mvn import MultivariateNormalReturns, StudentTReturns
from monteplan.models.returns.regime_switching import RegimeSwitchingReturns
from monteplan.models.stress import apply_stress_scenarios
from monteplan.policies.contributions import apply_contributions, compute_monthly_contributions
from monteplan.policies.rebalancing import rebalance_if_drifted, rebalance_to_targets
from monteplan.policies.spending.constant_real import ConstantRealSpending
from monteplan.policies.spending.floor_ceiling import FloorCeilingSpending
from monteplan.policies.spending.guardrails import GuardrailsSpending
from monteplan.policies.spending.percent_of_portfolio import PercentOfPortfolioSpending
from monteplan.policies.spending.vpw import VPWSpending
from monteplan.policies.withdrawals import withdraw
from monteplan.taxes.rmd import RMDCalculator
from monteplan.taxes.simple import FlatTaxModel
from monteplan.taxes.us_federal import USFederalTaxModel


@dataclass
class SimulationResult:
    """Output of a simulation run."""

    success_probability: float
    terminal_wealth_percentiles: dict[str, float]
    wealth_time_series: dict[str, np.ndarray]
    spending_time_series: dict[str, np.ndarray]
    n_paths: int
    n_steps: int
    seed: int
    plan: PlanConfig
    market: MarketAssumptions
    policies: PolicyBundle
    sim_config: SimulationConfig
    config_hash: str = ""
    engine_version: str = ""
    all_paths: np.ndarray | None = field(default=None, repr=False)
    inflation_rates: np.ndarray | None = field(default=None, repr=False)
    spending_history: np.ndarray | None = field(default=None, repr=False)


def simulate(
    plan: PlanConfig,
    market: MarketAssumptions,
    policies: PolicyBundle,
    sim_config: SimulationConfig,
) -> SimulationResult:
    """Run a Monte Carlo simulation.

    Args:
        plan: Financial plan configuration.
        market: Market return and inflation assumptions.
        policies: Spending, withdrawal, and rebalancing policies.
        sim_config: Simulation execution parameters.

    Returns:
        SimulationResult with success probability, percentiles, and time series.
    """
    rng = make_rng(sim_config.seed)
    n_paths = sim_config.n_paths
    antithetic = sim_config.antithetic

    # Round n_paths to even when antithetic is enabled
    if antithetic and n_paths % 2 != 0:
        n_paths += 1

    # Build timeline
    timeline = Timeline.from_ages(
        plan.current_age,
        plan.retirement_age,
        plan.end_age,
        plan.income_end_age,
    )
    n_steps = timeline.n_steps

    # Pre-generate all returns and inflation upfront
    if market.return_model == "regime_switching":
        if market.regime_switching is None:
            raise ValueError("regime_switching config must be provided for regime_switching model")
        rs_model = RegimeSwitchingReturns(market.regime_switching, antithetic=antithetic)
        returns = rs_model.sample(n_paths, n_steps, rng)

        # Coupled inflation model using regime indices
        rs_inflation = RegimeSwitchingInflationModel(
            regimes=list(market.regime_switching.regimes),
        )
        assert rs_model.regime_indices is not None
        inflation_rates = rs_inflation.sample(n_paths, n_steps, rng, rs_model.regime_indices)
    else:
        if market.return_model == "student_t":
            return_model_obj: (
                MultivariateNormalReturns | StudentTReturns | HistoricalBootstrapReturns
            ) = StudentTReturns(market, antithetic=antithetic)
        elif market.return_model == "bootstrap":
            if market.historical_returns is None:
                raise ValueError("historical_returns must be provided for bootstrap model")
            # Bootstrap: antithetic not applicable (no normal draws), silently ignored
            return_model_obj = HistoricalBootstrapReturns(
                np.array(market.historical_returns),
                block_size=market.bootstrap_block_size,
            )
        else:
            return_model_obj = MultivariateNormalReturns(market, antithetic=antithetic)
        returns = return_model_obj.sample(n_paths, n_steps, rng)  # (n_paths, n_steps, n_assets)

        inflation_model = OUInflationModel(
            theta=market.inflation_mean,
            sigma=market.inflation_vol,
            antithetic=antithetic,
        )
        inflation_rates = inflation_model.sample(n_paths, n_steps, rng)  # (n_paths, n_steps)

    # Apply stress scenario overlays (before main loop)
    if sim_config.stress_scenarios:
        returns, inflation_rates = apply_stress_scenarios(
            returns,
            inflation_rates,
            sim_config.stress_scenarios,
            timeline,
        )

    # Target asset weights (static or glide path per account)
    n_accounts = len(plan.accounts)
    allocations = market.asset_allocations
    if len(allocations) == 1 and n_accounts > 1:
        allocations = allocations * n_accounts
    elif len(allocations) != n_accounts:
        raise ValueError(f"Expected {n_accounts} or 1 asset allocations, got {len(allocations)}")

    static_weights = np.zeros((n_accounts, len(allocations[0].assets)))
    for i, alloc in enumerate(allocations):
        static_weights[i, :] = [a.weight for a in alloc.assets]

    # Pre-extract glide path information
    has_glide_path = any(a.glide_path is not None for a in allocations)
    glide_paths = [a.glide_path for a in allocations]
    gp_start_weights_list = [np.array(gp.start_weights) if gp else None for gp in glide_paths]
    gp_end_weights_list = [np.array(gp.end_weights) if gp else None for gp in glide_paths]

    weights = static_weights.copy()

    # Initialize state with per-asset-per-account positions
    initial_balances = [a.balance for a in plan.accounts]
    account_types: list[str] = [a.account_type for a in plan.accounts]
    state = SimulationState.initialize(n_paths, initial_balances, account_types, weights)
    state.initial_portfolio_value = state.total_wealth.copy()

    # Prepare policies
    monthly_contributions = compute_monthly_contributions(
        [a.annual_contribution for a in plan.accounts]
    )

    spending_policy: (
        ConstantRealSpending
        | PercentOfPortfolioSpending
        | GuardrailsSpending
        | VPWSpending
        | FloorCeilingSpending
    )
    if policies.spending.policy_type == "constant_real":
        spending_policy = ConstantRealSpending(plan.monthly_spending)
    elif policies.spending.policy_type == "percent_of_portfolio":
        spending_policy = PercentOfPortfolioSpending(policies.spending.withdrawal_rate)
    elif policies.spending.policy_type == "guardrails":
        spending_policy = GuardrailsSpending(policies.spending.guardrails)
    elif policies.spending.policy_type == "floor_ceiling":
        spending_policy = FloorCeilingSpending(policies.spending.floor_ceiling)
    else:
        spending_policy = VPWSpending(policies.spending.vpw, plan.end_age, plan.current_age)

    # Build tax model
    tax_model: FlatTaxModel | USFederalTaxModel
    if policies.tax_model == "us_federal":
        tax_model = USFederalTaxModel()
    else:
        tax_model = FlatTaxModel(policies.tax_rate)
    effective_tax_rate = tax_model.tax_rate_traditional() + policies.state_tax_rate
    rmd_calc = RMDCalculator()

    # Pre-compute discrete event steps
    event_steps: dict[int, float] = {}
    for ev in plan.discrete_events:
        ev_step = int(round((ev.age - plan.current_age) * 12))
        if 0 <= ev_step < n_steps:
            event_steps[ev_step] = event_steps.get(ev_step, 0.0) + ev.amount

    # Pre-compute guaranteed income streams (start/end steps and monthly amounts)
    gi_streams: list[tuple[int, int, float, float]] = []
    for gi in plan.guaranteed_income:
        gi_start = int(round((gi.start_age - plan.current_age) * 12))
        gi_end = int(round((gi.end_age - plan.current_age) * 12)) if gi.end_age else n_steps
        gi_monthly_cola = (1.0 + gi.cola_rate) ** (1.0 / 12.0) - 1.0
        gi_streams.append((gi_start, gi_end, gi.monthly_amount, gi_monthly_cola))

    # Pre-compute Roth conversion window
    roth_cfg = policies.roth_conversion
    roth_enabled = roth_cfg.enabled
    roth_start_step = -1
    roth_end_step = -1
    roth_trad_indices: list[int] = []
    roth_acct_idx = -1
    roth_bracket_ceiling = 0.0
    if roth_enabled:
        roth_start_step = int(round((roth_cfg.start_age - plan.current_age) * 12))
        roth_end_step = int(round((roth_cfg.end_age - plan.current_age) * 12))
        roth_trad_indices = [i for i, tp in enumerate(account_types) if tp == "traditional"]
        roth_indices = [i for i, tp in enumerate(account_types) if tp == "roth"]
        roth_acct_idx = roth_indices[0] if roth_indices else -1
        if roth_cfg.strategy == "fill_bracket" and isinstance(tax_model, USFederalTaxModel):
            roth_bracket_ceiling = tax_model.bracket_ceiling(
                roth_cfg.fill_to_bracket_top, policies.filing_status
            )

    # Investment fee drag (annualized → monthly)
    total_annual_fee = market.expense_ratio + market.aum_fee + market.advisory_fee
    monthly_fee = total_annual_fee / 12.0

    # Income growth tracking
    monthly_income_growth_rate = (1.0 + plan.income_growth_rate) ** (1.0 / 12.0) - 1.0
    income_growth_factor = 1.0

    # Storage for wealth and spending time series
    wealth_history = np.empty((n_paths, n_steps + 1))
    wealth_history[:, 0] = state.total_wealth
    spending_history = np.zeros((n_paths, n_steps))

    # Main simulation loop
    for t in range(n_steps):
        state.step = t

        # 1. Apply per-asset returns to positions
        asset_returns = returns[:, t, :]  # (n_paths, n_assets)
        asset_growth = 1.0 + asset_returns  # (n_paths, n_assets)
        state.positions *= asset_growth[:, np.newaxis, :]

        # Ensure no negative positions
        np.maximum(state.positions, 0.0, out=state.positions)

        # 1b. Apply investment fee drag (monthly deduction)
        if total_annual_fee > 0:
            state.positions *= 1.0 - monthly_fee

        # 2. Update cumulative inflation
        state.cumulative_inflation *= 1.0 + inflation_rates[:, t]

        # Compute current target weights (glide path or static)
        if has_glide_path:
            age = timeline.age_at(t)
            current_weights = np.zeros_like(static_weights)
            for i in range(n_accounts):
                gp = glide_paths[i]
                start_w = gp_start_weights_list[i]
                end_w = gp_end_weights_list[i]
                if gp is not None and start_w is not None and end_w is not None:
                    if age <= gp.start_age:
                        current_weights[i, :] = start_w
                    elif age >= gp.end_age:
                        current_weights[i, :] = end_w
                    else:
                        frac = (age - gp.start_age) / (gp.end_age - gp.start_age)
                        current_weights[i, :] = start_w + frac * (end_w - start_w)
                else:
                    current_weights[i, :] = static_weights[i, :]
        else:
            current_weights = static_weights

        # 3. Accumulation phase: add contributions with income growth
        if timeline.has_income(t):
            apply_contributions(
                state,
                monthly_contributions,
                current_weights,
                income_growth_factor,
            )
            income_growth_factor *= 1.0 + monthly_income_growth_rate

        # 4. Apply discrete events
        if t in event_steps:
            event_amount = event_steps[t]
            if event_amount > 0:
                # Inflow: distribute pro-rata across accounts by current balance
                total = state.total_wealth
                safe_total = np.where(total > 0, total, 1.0)
                for acct_idx in range(state.n_accounts):
                    acct_bal = state.balances[:, acct_idx]
                    share = acct_bal / safe_total
                    state.positions[:, acct_idx, :] += (
                        event_amount
                        * share[:, np.newaxis]
                        * current_weights[acct_idx][np.newaxis, :]
                    )
            else:
                # Outflow: withdraw pro-rata across accounts
                withdraw(
                    state,
                    np.full(n_paths, -event_amount),
                    list(policies.withdrawal_order),
                    effective_tax_rate,
                )

        # 5. Rebalancing (calendar or threshold-based)
        month = timeline.month_of_year(t)
        if policies.rebalancing_strategy == "threshold":
            rebalance_if_drifted(state, current_weights, policies.rebalancing_threshold)
        elif month in policies.rebalancing_months:
            rebalance_to_targets(state, current_weights)

        # 6. Retirement phase: compute spending, withdraw, apply taxes
        if timeline.is_retired(t):
            spending_need = spending_policy.compute(state)

            # Record total spending (before GI offset, for time series)
            spending_history[:, t] = spending_need

            # Subtract guaranteed income streams from spending need
            for gi_start, gi_end, gi_amount, gi_cola in gi_streams:
                if gi_start <= t < gi_end:
                    months_active = t - gi_start
                    cola_factor = (1.0 + gi_cola) ** months_active
                    gi_nominal = gi_amount * cola_factor * state.cumulative_inflation
                    spending_need = np.maximum(spending_need - gi_nominal, 0.0)

            # Only withdraw for non-depleted paths
            spending_need = np.where(state.is_depleted, 0.0, spending_need)

            # Track traditional balance before withdrawal to compute ordinary income
            trad_indices = [i for i, tp in enumerate(state.account_types) if tp == "traditional"]
            trad_before = (
                sum(state.balances[:, i] for i in trad_indices)
                if trad_indices
                else np.zeros(n_paths)
            )

            withdrawal_order: list[str] = list(policies.withdrawal_order)
            withdraw(
                state,
                spending_need,
                withdrawal_order,
                effective_tax_rate,
            )

            # Accumulate ordinary income from traditional withdrawals
            if trad_indices:
                trad_after = sum(state.balances[:, i] for i in trad_indices)
                trad_withdrawn = trad_before - trad_after
                state.annual_ordinary_income += np.maximum(trad_withdrawn, 0.0)

            # Track traditional withdrawn for RMD satisfaction
            if trad_indices:
                state.annual_rmd_satisfied += np.maximum(trad_withdrawn, 0.0)

            # RMD enforcement: force additional withdrawals if RMD not met
            age_int = int(timeline.age_at(t))
            if age_int >= rmd_calc.start_age and trad_indices and month == 12:
                rmd_required = rmd_calc.compute_rmd(
                    age_int,
                    state.prior_year_traditional_balance,
                )
                rmd_shortfall = np.maximum(rmd_required - state.annual_rmd_satisfied, 0.0)
                mask = rmd_shortfall > 0
                if mask.any():
                    # Force withdraw from traditional only
                    for idx in trad_indices:
                        available = state.positions[:, idx, :].sum(axis=1)
                        force_withdraw = np.minimum(rmd_shortfall, available)
                        force_withdraw = np.where(mask, force_withdraw, 0.0)
                        safe_avail = np.where(available > 0, available, 1.0)
                        frac_arr = np.minimum(force_withdraw / safe_avail, 1.0)
                        state.positions[:, idx, :] *= 1.0 - frac_arr[:, np.newaxis]
                        state.annual_ordinary_income += force_withdraw
                        rmd_shortfall -= force_withdraw

        # 6b. Roth conversions (year-end, within conversion window)
        if (
            roth_enabled
            and roth_trad_indices
            and roth_acct_idx >= 0
            and month == 12
            and roth_start_step <= t < roth_end_step
        ):
            # Determine conversion amount per path
            if roth_cfg.strategy == "fill_bracket":
                # Fill to bracket ceiling minus already-accumulated ordinary income
                conversion_target = np.maximum(
                    roth_bracket_ceiling - state.annual_ordinary_income, 0.0
                )
            else:
                conversion_target = np.full(n_paths, roth_cfg.annual_amount)

            conversion_target = np.where(state.is_depleted, 0.0, conversion_target)

            # Collect available traditional balance
            trad_available = np.zeros(n_paths)
            for idx in roth_trad_indices:
                trad_available += state.positions[:, idx, :].sum(axis=1)

            # Actual conversion = min(target, available)
            actual_conversion = np.minimum(conversion_target, trad_available)

            # Move positions pro-rata from traditional to first Roth account
            convert_mask = actual_conversion > 0
            if convert_mask.any():
                remaining = actual_conversion.copy()
                for idx in roth_trad_indices:
                    acct_bal = state.positions[:, idx, :].sum(axis=1)
                    move = np.minimum(remaining, acct_bal)
                    safe_bal = np.where(acct_bal > 0, acct_bal, 1.0)
                    frac = np.minimum(move / safe_bal, 1.0)
                    # Assets to move (pro-rata within account)
                    moved_positions = state.positions[:, idx, :] * frac[:, np.newaxis]
                    state.positions[:, idx, :] -= moved_positions
                    state.positions[:, roth_acct_idx, :] += moved_positions
                    remaining -= move

                # Conversion counts as ordinary income
                state.annual_ordinary_income += actual_conversion

        # 7. Year-end annual tax computation (for US federal model)
        if policies.tax_model == "us_federal" and month == 12:
            # Vectorized tax computation across all paths
            annual_tax = tax_model.compute_annual_tax_vectorized(
                state.annual_ordinary_income,
                state.annual_ltcg,
                policies.filing_status,
            )
            # State income tax overlay
            if policies.state_tax_rate > 0:
                annual_tax = (
                    annual_tax
                    + (state.annual_ordinary_income + state.annual_ltcg) * policies.state_tax_rate
                )
            # Net Investment Income Tax (3.8% surtax)
            if policies.include_niit and isinstance(tax_model, USFederalTaxModel):
                annual_tax = annual_tax + tax_model.compute_niit_vectorized(
                    state.annual_ordinary_income,
                    state.annual_ltcg,
                    policies.filing_status,
                )
            annual_tax = np.where(state.is_depleted, 0.0, annual_tax)
            total_w = state.total_wealth
            safe_total = np.where(total_w > 0, total_w, 1.0)
            tax_frac = np.minimum(annual_tax / safe_total, 1.0)
            state.positions *= 1.0 - tax_frac[:, np.newaxis, np.newaxis]
            # Snapshot traditional balance for next year's RMD
            trad_indices_yr = [
                i for i, tp in enumerate(state.account_types) if tp == "traditional"
            ]
            if trad_indices_yr:
                trad_sum = np.zeros(n_paths)
                for i in trad_indices_yr:
                    trad_sum += state.balances[:, i]
                state.prior_year_traditional_balance = trad_sum

            # Reset annual accumulators
            state.annual_ordinary_income[:] = 0.0
            state.annual_ltcg[:] = 0.0
            state.annual_rmd_satisfied[:] = 0.0

        # 8. Update depletion status
        state.is_depleted = state.total_wealth <= 0.0

        # 9. Record wealth snapshot
        wealth_history[:, t + 1] = state.total_wealth

    # Compute results
    # Success = portfolio never hit zero (check all retirement steps)
    retirement_wealth = wealth_history[:, timeline.retirement_step :]
    success_mask = np.all(retirement_wealth > 0, axis=1)
    success_probability = float(success_mask.mean())

    # Terminal wealth
    terminal_wealth = wealth_history[:, -1]
    percentile_keys = [5, 25, 50, 75, 95]
    terminal_pcts = np.percentile(terminal_wealth, percentile_keys)
    terminal_wealth_percentiles = {
        f"p{p}": float(v) for p, v in zip(percentile_keys, terminal_pcts, strict=True)
    }

    # Wealth time series percentiles (for fan charts)
    wealth_ts: dict[str, np.ndarray] = {}
    for p in percentile_keys:
        wealth_ts[f"p{p}"] = np.percentile(wealth_history, p, axis=0)

    wealth_ts["mean"] = wealth_history.mean(axis=0)

    # Spending time series percentiles (for spending fan charts)
    spending_ts: dict[str, np.ndarray] = {}
    for p in percentile_keys:
        spending_ts[f"p{p}"] = np.percentile(spending_history, p, axis=0)
    spending_ts["mean"] = spending_history.mean(axis=0)

    return SimulationResult(
        success_probability=success_probability,
        terminal_wealth_percentiles=terminal_wealth_percentiles,
        wealth_time_series=wealth_ts,
        spending_time_series=spending_ts,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=sim_config.seed,
        plan=plan,
        market=market,
        policies=policies,
        sim_config=sim_config,
        config_hash=compute_config_hash(plan, market, policies, sim_config),
        engine_version=__version__,
        all_paths=wealth_history if sim_config.store_paths else None,
        inflation_rates=inflation_rates if sim_config.store_paths else None,
        spending_history=spending_history if sim_config.store_paths else None,
    )
