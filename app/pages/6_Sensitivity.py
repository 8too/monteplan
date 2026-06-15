"""Sensitivity Analysis page — tornado chart and 2D heatmap."""

from __future__ import annotations

import streamlit as st

from monteplan.analytics.sensitivity import (
    _build_param_registry,
    run_2d_sensitivity,
    run_sensitivity,
)
from monteplan.config.defaults import (
    default_market,
    default_plan,
    default_policies,
    default_sim_config,
)
from monteplan.config.schema import (
    MarketAssumptions,
    PlanConfig,
    PolicyBundle,
    SimulationConfig,
)

st.set_page_config(page_title="Sensitivity — MontePlan", layout="wide")

from app.components.theme import register_theme

register_theme()

st.title("Sensitivity Analysis")

# Get configs from session state
plan: PlanConfig = st.session_state.get("plan", default_plan())
market: MarketAssumptions = st.session_state.get("market", default_market())
policies: PolicyBundle = st.session_state.get("policies", default_policies())
sim_config: SimulationConfig = st.session_state.get("sim_config", default_sim_config())

# --- Tornado Analysis ---
st.subheader("Tornado Analysis (One-at-a-Time)")

# Controls
col1, col2 = st.columns(2)
with col1:
    perturbation_pct = st.slider(
        "Perturbation (%)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="How much to vary each parameter (+/-%)",
    )

# Build default parameter list
n_assets = len(market.asset_allocations[0].assets)
default_params: list[str] = []
for i in range(n_assets):
    default_params.append(f"{market.asset_allocations[0].assets[i].name} Return")
    default_params.append(f"{market.asset_allocations[0].assets[i].name} Volatility")
default_params.append("Inflation Rate")
default_params.append("Monthly Spending")
default_params.append("Retirement Age")
for acct in plan.accounts:
    if acct.annual_contribution > 0:
        default_params.append(f"{acct.account_type.title()} Contribution")

with col2:
    selected_params = st.multiselect(
        "Parameters to analyze",
        options=default_params,
        default=default_params,
    )

if st.button("Run Sensitivity Analysis", type="primary"):
    with st.spinner("Running sensitivity analysis..."):
        report = run_sensitivity(
            plan,
            market,
            policies,
            sim_config,
            perturbation_pct=perturbation_pct / 100.0,
            parameters=selected_params if selected_params else None,
        )
    st.session_state["sensitivity_report"] = report

if "sensitivity_report" in st.session_state:
    report = st.session_state["sensitivity_report"]

    st.write(f"Base Success Probability: **{report.base_success_probability:.1%}**")

    from app.components.charts import tornado_chart

    chart_data = [
        {
            "parameter_name": r.parameter_name,
            "low_success": r.low_success,
            "high_success": r.high_success,
            "low_value": r.low_value,
            "high_value": r.high_value,
        }
        for r in report.results
    ]
    fig = tornado_chart(chart_data, report.base_success_probability)
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    with st.expander("Detail Table"):
        table_data = []
        for r in sorted(report.results, key=lambda x: abs(x.impact), reverse=True):
            table_data.append(
                {
                    "Parameter": r.parameter_name,
                    "Base Value": f"{r.base_value:.4f}",
                    "Low Value": f"{r.low_value:.4f}",
                    "High Value": f"{r.high_value:.4f}",
                    "Low Success": f"{r.low_success:.1%}",
                    "High Success": f"{r.high_success:.1%}",
                    "Impact (pp)": f"{r.impact * 100:+.1f}",
                }
            )
        st.table(table_data)

# --- 2D Sensitivity Heatmap ---
st.divider()
st.subheader("2D Sensitivity Heatmap")
st.caption("Vary two parameters simultaneously and visualize success probability across a grid.")

# Available parameters for 2D analysis
all_params_registry = _build_param_registry(plan, market)
available_2d_params = list(all_params_registry.keys())

col1, col2 = st.columns(2)
with col1:
    default_x_idx = (
        available_2d_params.index("Monthly Spending")
        if "Monthly Spending" in available_2d_params
        else 0
    )
    x_param = st.selectbox(
        "X-axis parameter",
        available_2d_params,
        index=default_x_idx,
        key="heatmap_x_param",
    )
with col2:
    default_y_idx = (
        available_2d_params.index("Equity Allocation")
        if "Equity Allocation" in available_2d_params
        else min(1, len(available_2d_params) - 1)
    )
    y_param = st.selectbox(
        "Y-axis parameter",
        available_2d_params,
        index=default_y_idx,
        key="heatmap_y_param",
    )

if x_param == y_param:
    st.warning("Please select two different parameters.")
else:
    # Determine base values and ranges
    x_spec = all_params_registry[x_param]
    y_spec = all_params_registry[y_param]
    x_target = plan if x_spec.target == "plan" else market
    y_target = plan if y_spec.target == "plan" else market
    base_x_val = x_spec.getter(x_target)
    base_y_val = y_spec.getter(y_target)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_min = st.number_input(
            f"{x_param} min",
            value=base_x_val * 0.5,
            format="%.4f",
            key="hm_x_min",
        )
    with col2:
        x_max = st.number_input(
            f"{x_param} max",
            value=base_x_val * 1.5,
            format="%.4f",
            key="hm_x_max",
        )
    with col3:
        y_min = st.number_input(
            f"{y_param} min",
            value=base_y_val * 0.5,
            format="%.4f",
            key="hm_y_min",
        )
    with col4:
        y_max = st.number_input(
            f"{y_param} max",
            value=base_y_val * 1.5,
            format="%.4f",
            key="hm_y_max",
        )

    if st.button("Run 2D Analysis"):
        progress = st.progress(0, text="Running 2D sensitivity...")
        heatmap_result = run_2d_sensitivity(
            plan,
            market,
            policies,
            sim_config,
            x_param=x_param,
            y_param=y_param,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            max_workers=1,
        )
        progress.progress(100, text="Done!")

        st.session_state["heatmap_data"] = {
            "x_param_name": heatmap_result.x_param_name,
            "y_param_name": heatmap_result.y_param_name,
            "x_values": heatmap_result.x_values,
            "y_values": heatmap_result.y_values,
            "success_grid": heatmap_result.success_grid,
            "base_x_value": heatmap_result.base_x_value,
            "base_y_value": heatmap_result.base_y_value,
            "base_success": heatmap_result.base_success,
        }

    if "heatmap_data" in st.session_state:
        from app.components.charts import sensitivity_heatmap

        heatmap_fig = sensitivity_heatmap(st.session_state["heatmap_data"])
        st.plotly_chart(heatmap_fig, use_container_width=True)
