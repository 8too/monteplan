"""Policies page — spending, tax, withdrawal ordering, and rebalancing."""

from __future__ import annotations

import streamlit as st

from monteplan.config.defaults import default_policies
from monteplan.config.schema import (
    FloorCeilingConfig,
    GuardrailsConfig,
    PolicyBundle,
    RothConversionConfig,
    SpendingPolicyConfig,
    VPWConfig,
)

st.set_page_config(page_title="Policies — MontePlan", layout="wide")
st.title("Policies")

# Initialize defaults
if "policies" not in st.session_state:
    st.session_state["policies"] = default_policies()

policies: PolicyBundle = st.session_state["policies"]

# --- Spending Policy ---
st.subheader("Spending Policy")

policy_options = ["constant_real", "percent_of_portfolio", "guardrails", "vpw", "floor_ceiling"]
policy_labels = {
    "constant_real": "Constant Real (inflation-adjusted flat amount)",
    "percent_of_portfolio": "Percent of Portfolio (fixed annual withdrawal rate)",
    "guardrails": "Guyton-Klinger Guardrails (adaptive with spending cuts/raises)",
    "vpw": "Variable Percentage Withdrawal (rate varies by remaining years)",
    "floor_ceiling": "Floor & Ceiling (percent of portfolio with min/max bounds)",
}
policy_type = st.selectbox(
    "Spending Policy",
    policy_options,
    index=policy_options.index(policies.spending.policy_type),
    format_func=lambda x: policy_labels.get(x, x),
)

withdrawal_rate = policies.spending.withdrawal_rate
if policy_type == "percent_of_portfolio":
    withdrawal_rate = st.number_input(
        "Annual Withdrawal Rate (%)",
        value=policies.spending.withdrawal_rate * 100,
        step=0.5,
        min_value=0.1,
        max_value=100.0,
    ) / 100

# Guardrails config
guardrails = policies.spending.guardrails
if policy_type == "guardrails":
    st.markdown("**Guardrails Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        gr_iwr = st.number_input(
            "Initial Withdrawal Rate (%)",
            value=guardrails.initial_withdrawal_rate * 100,
            step=0.5, min_value=1.0, max_value=20.0,
        ) / 100
        gr_upper = st.number_input(
            "Prosperity Threshold (%)",
            value=guardrails.upper_threshold * 100,
            step=5.0, min_value=5.0, max_value=100.0,
            help="If current rate drops below initial * (1 - threshold), increase spending",
        ) / 100
        gr_raise = st.number_input(
            "Raise Percentage (%)",
            value=guardrails.raise_pct * 100,
            step=1.0, min_value=1.0, max_value=100.0,
        ) / 100
    with col2:
        gr_lower = st.number_input(
            "Capital Preservation Threshold (%)",
            value=guardrails.lower_threshold * 100,
            step=5.0, min_value=5.0, max_value=100.0,
            help="If current rate exceeds initial * (1 + threshold), decrease spending",
        ) / 100
        gr_cut = st.number_input(
            "Cut Percentage (%)",
            value=guardrails.cut_pct * 100,
            step=1.0, min_value=1.0, max_value=100.0,
        ) / 100
    guardrails = GuardrailsConfig(
        initial_withdrawal_rate=gr_iwr,
        upper_threshold=gr_upper,
        lower_threshold=gr_lower,
        raise_pct=gr_raise,
        cut_pct=gr_cut,
    )

# VPW config
vpw = policies.spending.vpw
if policy_type == "vpw":
    st.markdown("**VPW Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        vpw_min = st.number_input(
            "Minimum Withdrawal Rate (%)",
            value=vpw.min_rate * 100,
            step=0.5, min_value=0.0, max_value=100.0,
        ) / 100
    with col2:
        vpw_max = st.number_input(
            "Maximum Withdrawal Rate (%)",
            value=vpw.max_rate * 100,
            step=0.5, min_value=0.0, max_value=100.0,
        ) / 100
    vpw = VPWConfig(min_rate=vpw_min, max_rate=vpw_max)

# Floor & Ceiling config
floor_ceiling = policies.spending.floor_ceiling
if policy_type == "floor_ceiling":
    st.markdown("**Floor & Ceiling Parameters**")
    col1, col2, col3 = st.columns(3)
    with col1:
        fc_rate = st.number_input(
            "Withdrawal Rate (%)",
            value=floor_ceiling.withdrawal_rate * 100,
            step=0.5, min_value=0.5, max_value=20.0,
        ) / 100
    with col2:
        fc_floor = st.number_input(
            "Monthly Floor ($)",
            value=floor_ceiling.floor,
            step=500.0, min_value=0.0,
            help="Minimum monthly spending in today's dollars (grows with inflation)",
        )
    with col3:
        fc_ceiling = st.number_input(
            "Monthly Ceiling ($)",
            value=floor_ceiling.ceiling,
            step=500.0, min_value=0.0,
            help="Maximum monthly spending in today's dollars (grows with inflation)",
        )
    floor_ceiling = FloorCeilingConfig(
        withdrawal_rate=fc_rate, floor=fc_floor, ceiling=fc_ceiling,
    )

# --- Tax Model ---
st.subheader("Tax Model")

tax_model_options = ["flat", "us_federal"]
tax_model_labels = {
    "flat": "Flat Rate",
    "us_federal": "US Federal Brackets (2026)",
}
tax_model = st.selectbox(
    "Tax Model",
    tax_model_options,
    index=tax_model_options.index(policies.tax_model),
    format_func=lambda x: tax_model_labels.get(x, x),
)

tax_rate = policies.tax_rate
filing_status = policies.filing_status

if tax_model == "flat":
    tax_rate = st.number_input(
        "Effective Tax Rate (%)",
        value=policies.tax_rate * 100,
        step=1.0, min_value=0.0, max_value=100.0,
    ) / 100

if tax_model == "us_federal":
    filing_status = st.selectbox(
        "Filing Status",
        ["single", "married_jointly"],
        index=["single", "married_jointly"].index(policies.filing_status),
        format_func=lambda x: x.replace("_", " ").title(),
    )

include_niit = False
if tax_model == "us_federal":
    include_niit = st.checkbox(
        "Include NIIT (3.8% Net Investment Income Tax)",
        value=policies.include_niit,
        help="3.8% surtax on investment income above $200K (single) / $250K (married)",
    )

state_tax_rate = st.number_input(
    "State Income Tax Rate (%)",
    value=policies.state_tax_rate * 100,
    step=0.5,
    min_value=0.0,
    max_value=15.0,
    help="Flat state income tax rate applied to ordinary income + LTCG (0 = no state tax)",
) / 100

# --- Roth Conversions ---
st.subheader("Roth Conversions")
st.caption(
    "Move money from traditional to Roth accounts, paying ordinary income tax now "
    "for tax-free future growth."
)

roth_cfg = policies.roth_conversion
roth_enabled = st.checkbox("Enable Roth Conversions", value=roth_cfg.enabled)

roth_strategy = roth_cfg.strategy
roth_annual_amount = roth_cfg.annual_amount
roth_fill_bracket = roth_cfg.fill_to_bracket_top
roth_start_age = roth_cfg.start_age
roth_end_age = roth_cfg.end_age

if roth_enabled:
    roth_strategy = st.selectbox(
        "Conversion Strategy",
        ["fixed_amount", "fill_bracket"],
        index=["fixed_amount", "fill_bracket"].index(roth_cfg.strategy),
        format_func=lambda x: {
            "fixed_amount": "Fixed Annual Amount",
            "fill_bracket": "Fill to Tax Bracket",
        }.get(x, x),
    )
    col1, col2 = st.columns(2)
    with col1:
        if roth_strategy == "fixed_amount":
            roth_annual_amount = st.number_input(
                "Annual Conversion Amount ($)",
                value=roth_cfg.annual_amount,
                step=5000.0,
                min_value=0.0,
                help="Fixed dollar amount to convert annually",
            )
        else:
            roth_fill_bracket = st.number_input(
                "Fill to Bracket Top (%)",
                value=roth_cfg.fill_to_bracket_top * 100,
                step=1.0,
                min_value=10.0,
                max_value=37.0,
                help="Convert enough to fill up to this marginal tax bracket",
            ) / 100
    with col2:
        roth_start_age = st.number_input(
            "Start Age", value=int(roth_cfg.start_age), min_value=18, max_value=120,
        )
        roth_end_age = st.number_input(
            "End Age", value=int(roth_cfg.end_age), min_value=18, max_value=120,
        )

# --- Withdrawal Order ---
st.subheader("Withdrawal Order")
st.caption("Priority order for withdrawing from accounts in retirement (first = withdrawn first)")

default_order = list(policies.withdrawal_order)
order_options = ["taxable", "traditional", "roth"]

w1 = st.selectbox("1st Priority", order_options, index=order_options.index(default_order[0]), key="wo_1")
remaining_1 = [x for x in order_options if x != w1]
w2 = st.selectbox("2nd Priority", remaining_1, index=0, key="wo_2")
remaining_2 = [x for x in remaining_1 if x != w2]
w3 = remaining_2[0]
st.write(f"3rd Priority: {w3.title()}")

# --- Rebalancing ---
st.subheader("Rebalancing")
rebal_strategy_options = ["calendar", "threshold"]
rebal_strategy_labels = {
    "calendar": "Calendar (fixed schedule)",
    "threshold": "Threshold (rebalance when drift exceeds limit)",
}
rebalancing_strategy = st.selectbox(
    "Rebalancing Strategy",
    rebal_strategy_options,
    index=rebal_strategy_options.index(policies.rebalancing_strategy),
    format_func=lambda x: rebal_strategy_labels.get(x, x),
)

rebalancing_months = policies.rebalancing_months
rebalancing_threshold = policies.rebalancing_threshold

if rebalancing_strategy == "calendar":
    rebal_options = {
        "Monthly": list(range(1, 13)),
        "Quarterly": [1, 4, 7, 10],
        "Semi-Annual": [1, 7],
        "Annual": [1],
    }
    rebal_choice = st.selectbox(
        "Rebalancing Frequency",
        list(rebal_options.keys()),
        index=list(rebal_options.keys()).index(
            next(
                (k for k, v in rebal_options.items() if v == policies.rebalancing_months),
                "Semi-Annual",
            )
        ),
    )
    rebalancing_months = rebal_options[rebal_choice]
else:
    rebalancing_threshold = st.number_input(
        "Drift Threshold (%)",
        value=policies.rebalancing_threshold * 100,
        step=1.0, min_value=1.0, max_value=50.0,
        help="Rebalance when any asset drifts more than this percentage from target",
    ) / 100

# --- Save ---
if st.button("Save Policies", type="primary"):
    try:
        new_policies = PolicyBundle(
            spending=SpendingPolicyConfig(
                policy_type=policy_type,
                withdrawal_rate=withdrawal_rate,
                guardrails=guardrails,
                vpw=vpw,
                floor_ceiling=floor_ceiling,
            ),
            tax_model=tax_model,
            tax_rate=tax_rate,
            filing_status=filing_status,
            state_tax_rate=state_tax_rate,
            include_niit=include_niit,
            roth_conversion=RothConversionConfig(
                enabled=roth_enabled,
                strategy=roth_strategy,
                annual_amount=roth_annual_amount,
                fill_to_bracket_top=roth_fill_bracket,
                start_age=roth_start_age,
                end_age=roth_end_age,
            ),
            withdrawal_order=[w1, w2, w3],
            rebalancing_strategy=rebalancing_strategy,
            rebalancing_months=rebalancing_months,
            rebalancing_threshold=rebalancing_threshold,
        )
        st.session_state["policies"] = new_policies
        st.success("Policies saved!")
    except Exception as e:
        st.error(f"Validation error: {e}")
