"""Portfolio page — asset allocation, market assumptions, return model, stress scenarios."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import app.components.charts as app_charts
import app.components.theme as app_theme
import monteplan.config.defaults as mp_defaults
import monteplan.config.schema as mp_schema

st.set_page_config(page_title="Portfolio — MontePlan", layout="wide")

app_theme.register_theme()

st.title("Portfolio & Assumptions")

# Initialize defaults
if "market" not in st.session_state:
    st.session_state["market"] = mp_defaults.default_market()
if "sim_config" not in st.session_state:
    st.session_state["sim_config"] = mp_defaults.default_sim_config()

market: mp_schema.MarketAssumptions = st.session_state["market"]
sim_config: mp_schema.SimulationConfig = st.session_state["sim_config"]
plan_data: mp_schema.PlanConfig = st.session_state.get("plan", mp_defaults.default_plan())


def infer_allocation_settings(alloc: mp_schema.AssetAllocation) -> dict:
    """Infer UI settings from an AssetAllocation object."""
    n_assets = len(alloc.assets)
    if n_assets == 2:
        preset = "US Only"
        stock_pct = int(round(alloc.assets[0].weight * 100))
        stock_us, stock_exus, stock_em = 100, 0, 0
        bond_us, bond_exus, bond_em = 100, 0, 0
        same_bond_split = True
        bond_type = "Aggregate (2% real)"
    else:
        stock_weights = [a.weight for a in alloc.assets if "Stock" in a.name]
        bond_weights = [a.weight for a in alloc.assets if "Bond" in a.name]

        sum_stock = sum(stock_weights)
        stock_pct = int(round(sum_stock * 100))

        if sum_stock > 1e-6:
            stock_us = int(round(stock_weights[0] / sum_stock * 100))
            stock_exus = int(round(stock_weights[1] / sum_stock * 100))
            stock_em = 100 - stock_us - stock_exus
        else:
            stock_us, stock_exus, stock_em = 60, 30, 10

        sum_bond = sum(bond_weights)
        if sum_bond > 1e-6:
            bond_us = int(round(bond_weights[0] / sum_bond * 100))
            bond_exus = int(round(bond_weights[1] / sum_bond * 100))
            bond_em = 100 - bond_us - bond_exus
        else:
            bond_us, bond_exus, bond_em = 60, 30, 10

        same_bond_split = (
            stock_us == bond_us and stock_exus == bond_exus and stock_em == bond_em
        )

        # Infer preset
        is_default_stock = stock_us == 60 and stock_exus == 30 and stock_em == 10
        is_default_bond = bond_us == 60 and bond_exus == 30 and bond_em == 10
        if is_default_stock and (same_bond_split or is_default_bond):
            preset = "Global Diversified"
        else:
            preset = "Custom"

        bond_type = "Aggregate (2% real)"

    use_glide = alloc.glide_path is not None
    gp_start_age = alloc.glide_path.start_age if use_glide else 30
    gp_end_age = alloc.glide_path.end_age if use_glide else 70

    if use_glide and alloc.glide_path:
        if n_assets == 6:
            gp_end_stock = int(round(sum(alloc.glide_path.end_weights[:3]) * 100))
        else:
            gp_end_stock = int(round(alloc.glide_path.end_weights[0] * 100))
    else:
        gp_end_stock = 40

    return {
        "preset": preset,
        "stock_pct": stock_pct,
        "stock_us": stock_us,
        "stock_exus": stock_exus,
        "stock_em": stock_em,
        "same_bond_split": same_bond_split,
        "bond_us": bond_us,
        "bond_exus": bond_exus,
        "bond_em": bond_em,
        "bond_type": bond_type,
        "use_glide": use_glide,
        "gp_start_age": gp_start_age,
        "gp_end_age": gp_end_age,
        "gp_end_stock": gp_end_stock,
    }


if "use_all_accounts" not in st.session_state:
    st.session_state["use_all_accounts"] = len(market.asset_allocations) == 1

if "alloc_settings" not in st.session_state:
    # Read from market.asset_allocations
    if len(market.asset_allocations) == 1:
        single_settings = infer_allocation_settings(market.asset_allocations[0])
        st.session_state["alloc_settings"] = {
            "taxable": dict(single_settings),
            "traditional": dict(single_settings),
            "roth": dict(single_settings),
        }
    else:
        default_settings = infer_allocation_settings(market.asset_allocations[0])
        st.session_state["alloc_settings"] = {
            "taxable": dict(default_settings),
            "traditional": dict(default_settings),
            "roth": dict(default_settings),
        }
        if len(market.asset_allocations) == len(plan_data.accounts):
            for account, alloc in zip(plan_data.accounts, market.asset_allocations):
                st.session_state["alloc_settings"][account.account_type] = infer_allocation_settings(alloc)

if "selected_account_type" not in st.session_state:
    st.session_state["selected_account_type"] = "Taxable"

# --- Portfolio Preset ---
st.subheader("Asset Allocation")

# Checkbox and Popup Menu
col_cb, col_pop = st.columns(2)
with col_cb:
    use_all_accounts = st.checkbox(
        "Use for all Accounts",
        value=st.session_state["use_all_accounts"],
        key="use_all_accounts_widget",
    )
    if use_all_accounts != st.session_state["use_all_accounts"]:
        st.session_state["use_all_accounts"] = use_all_accounts
        if not use_all_accounts:
            taxable_settings = dict(st.session_state["alloc_settings"]["taxable"])
            st.session_state["alloc_settings"]["traditional"] = dict(taxable_settings)
            st.session_state["alloc_settings"]["roth"] = dict(taxable_settings)
        st.rerun()

active_type = "taxable"
if not use_all_accounts:
    with col_pop:
        selected_account_type = st.selectbox(
            "Account Type",
            ["Taxable", "Traditional", "Roth"],
            index=["Taxable", "Traditional", "Roth"].index(
                st.session_state["selected_account_type"]
            ),
            key="account_type_widget",
        )
        if selected_account_type != st.session_state["selected_account_type"]:
            st.session_state["selected_account_type"] = selected_account_type
            st.rerun()
    active_type = st.session_state["selected_account_type"].lower()

settings = st.session_state["alloc_settings"][active_type]


def render_allocation_section(active_type: str, settings: dict):
    preset = st.radio(
        "Portfolio Preset",
        ["Global Diversified", "US Only", "Custom"],
        index=["Global Diversified", "US Only", "Custom"].index(settings["preset"]),
        horizontal=True,
        key=f"preset_{active_type}",
        help="Global: 6-asset US/Ex-US/EM. US Only: 2-asset US Stocks + US Bonds.",
    )
    settings["preset"] = preset

    is_global = preset == "Global Diversified"
    is_custom = preset == "Custom"

    stock_pct = st.slider(
        "Stocks (%)",
        min_value=0,
        max_value=100,
        value=settings["stock_pct"],
        step=5,
        key=f"stock_pct_{active_type}",
    )
    settings["stock_pct"] = stock_pct
    bond_pct = 100 - stock_pct
    st.write(f"Bonds: {bond_pct}%")

    stock_us, stock_exus, stock_em = (
        settings["stock_us"],
        settings["stock_exus"],
        settings["stock_em"],
    )
    bond_us, bond_exus, bond_em = settings["bond_us"], settings["bond_exus"], settings["bond_em"]
    same_bond_split = settings["same_bond_split"]

    if is_global or is_custom:
        st.markdown("**Regional Split (Stocks)**")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            stock_us = st.number_input(
                "US (%)",
                min_value=0,
                max_value=100,
                value=stock_us,
                step=5,
                key=f"stock_us_{active_type}",
            )
        with scol2:
            stock_exus = st.number_input(
                "Ex-US Dev (%)",
                min_value=0,
                max_value=100,
                value=stock_exus,
                step=5,
                key=f"stock_exus_{active_type}",
            )
        with scol3:
            stock_em = st.number_input(
                "EM (%)",
                min_value=0,
                max_value=100,
                value=stock_em,
                step=5,
                key=f"stock_em_{active_type}",
            )
        settings["stock_us"] = stock_us
        settings["stock_exus"] = stock_exus
        settings["stock_em"] = stock_em
        stock_reg_sum = stock_us + stock_exus + stock_em
        if stock_reg_sum != 100:
            st.warning(f"Stock regional split sums to {stock_reg_sum}%, must be 100%.")

        same_bond_split = st.checkbox(
            "Same regional split for bonds",
            value=same_bond_split,
            key=f"same_bond_split_{active_type}",
        )
        settings["same_bond_split"] = same_bond_split
        if same_bond_split:
            bond_us, bond_exus, bond_em = stock_us, stock_exus, stock_em
        else:
            st.markdown("**Regional Split (Bonds)**")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                bond_us = st.number_input(
                    "US (%)",
                    min_value=0,
                    max_value=100,
                    value=bond_us,
                    step=5,
                    key=f"bond_us_{active_type}",
                )
            with bcol2:
                bond_exus = st.number_input(
                    "Ex-US Dev (%)",
                    min_value=0,
                    max_value=100,
                    value=bond_exus,
                    step=5,
                    key=f"bond_exus_{active_type}",
                )
            with bcol3:
                bond_em = st.number_input(
                    "EM (%)",
                    min_value=0,
                    max_value=100,
                    value=bond_em,
                    step=5,
                    key=f"bond_em_{active_type}",
                )
            bond_reg_sum = bond_us + bond_exus + bond_em
            if bond_reg_sum != 100:
                st.warning(f"Bond regional split sums to {bond_reg_sum}%, must be 100%.")
        settings["bond_us"] = bond_us
        settings["bond_exus"] = bond_exus
        settings["bond_em"] = bond_em

        stock_regional = [stock_us / 100, stock_exus / 100, stock_em / 100]
        bond_regional = [bond_us / 100, bond_exus / 100, bond_em / 100]

        # --- Bond Type ---
        bond_type = st.radio(
            "Bond Type",
            ["Aggregate (2% real)", "Treasuries Only (1.5% real)"],
            index=0 if "Aggregate" in settings["bond_type"] else 1,
            horizontal=True,
            key=f"bond_type_{active_type}",
        )
        settings["bond_type"] = bond_type
        bond_type_key = "treasuries" if "Treasuries" in bond_type else "aggregate"

        # --- Computed Weights Display ---
        computed_weights = mp_defaults.build_global_weights(
            stock_pct=stock_pct / 100,
            stock_regional=stock_regional,
            bond_regional=bond_regional,
        )
        st.markdown("**Computed Weights**")
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            for i, name in enumerate(mp_defaults.STOCK_ASSET_NAMES):
                st.write(f"{name}: {computed_weights[i]:.0%}")
        with wcol2:
            for i, name in enumerate(mp_defaults.BOND_ASSET_NAMES):
                st.write(f"{name}: {computed_weights[i + 3]:.0%}")

        n_assets = 6
        asset_names = list(mp_defaults.ALL_ASSET_NAMES)
        asset_weights = computed_weights

    else:
        # US Only — simple 2-asset
        bond_type_key = "aggregate"
        n_assets = 2
        asset_names = ["US Stocks", "US Bonds"]
        asset_weights = [stock_pct / 100, bond_pct / 100]
        stock_regional = None
        bond_regional = None

    # --- Glide Path ---
    use_glide = st.checkbox(
        "Enable Glide Path (age-based allocation shift)",
        value=settings["use_glide"],
        key=f"use_glide_{active_type}",
    )
    settings["use_glide"] = use_glide

    gp_start_age = settings["gp_start_age"]
    gp_end_age = settings["gp_end_age"]
    gp_end_stock = settings["gp_end_stock"]

    if use_glide:
        col1, col2, col3 = st.columns(3)
        with col1:
            gp_start_age = st.number_input(
                "Glide Start Age",
                min_value=18,
                max_value=120,
                value=gp_start_age,
                key=f"gp_start_age_{active_type}",
            )
        with col2:
            gp_end_age = st.number_input(
                "Glide End Age",
                min_value=gp_start_age + 1,
                max_value=120,
                value=max(gp_end_age, gp_start_age + 1),
                key=f"gp_end_age_{active_type}",
            )
        with col3:
            gp_end_stock = st.slider(
                "End Stock Allocation (%)",
                min_value=0,
                max_value=100,
                step=5,
                value=gp_end_stock,
                key=f"gp_end_stock_{active_type}",
            )
        st.caption(
            f"Allocation shifts from {stock_pct}% stocks at age {gp_start_age} "
            f"to {gp_end_stock}% stocks at age {gp_end_age}"
        )
    settings["gp_start_age"] = gp_start_age
    settings["gp_end_age"] = gp_end_age
    settings["gp_end_stock"] = gp_end_stock

    # --- Allocation area chart preview ---
    _preview_current_age = plan_data.current_age if plan_data else 30
    _preview_end_age = plan_data.end_age if plan_data else 90
    _preview_assets = [
        {"name": asset_names[i], "weight": asset_weights[i]} for i in range(n_assets)
    ]
    _preview_gp = None
    if use_glide:
        if n_assets == 6:
            end_weights = mp_defaults.build_global_weights(
                stock_pct=gp_end_stock / 100,
                stock_regional=stock_regional,
                bond_regional=bond_regional,
            )
        else:
            end_weights = [gp_end_stock / 100, (100 - gp_end_stock) / 100]
        _preview_gp = {
            "start_age": gp_start_age,
            "start_weights": list(asset_weights),
            "end_age": gp_end_age,
            "end_weights": end_weights,
        }
    _alloc_fig = app_charts.allocation_area_chart(
        _preview_assets,
        _preview_gp,
        _preview_current_age,
        _preview_end_age,
    )
    st.plotly_chart(_alloc_fig, use_container_width=True)

    return (
        n_assets,
        asset_names,
        asset_weights,
        bond_type_key,
        preset,
        stock_regional,
        bond_regional,
    )


if use_all_accounts:
    (
        n_assets,
        asset_names,
        asset_weights,
        bond_type_key,
        preset,
        stock_regional,
        bond_regional,
    ) = render_allocation_section(active_type, settings)
else:
    with st.container(border=True):
        st.subheader(f"Asset Allocation for {st.session_state['selected_account_type']} Accounts")
        (
            n_assets,
            asset_names,
            asset_weights,
            bond_type_key,
            preset,
            stock_regional,
            bond_regional,
        ) = render_allocation_section(active_type, settings)

current_assets_count = len(market.asset_allocations[0].assets)


# --- Market Assumptions ---
with st.expander("Market Assumptions", expanded=False):
    if n_assets == 6:
        # Get reference market for default return/vol values
        ref_market = mp_defaults.default_market(bond_type=bond_type_key)

        # Use current market values if asset count matches, otherwise use reference
        use_current = current_assets_count == 6

        st.markdown("**Expected Returns (%/yr)**")
        ret_cols = st.columns(3)
        returns_input: list[float] = []
        for i in range(6):
            with ret_cols[i % 3]:
                default_val = (
                    market.expected_annual_returns[i] * 100
                    if use_current
                    else ref_market.expected_annual_returns[i] * 100
                )
                val = st.number_input(
                    f"{mp_defaults.ALL_ASSET_NAMES[i]}",
                    value=default_val,
                    step=0.5,
                    key=f"ret_{i}",
                )
                returns_input.append(val)

        st.markdown("**Volatility (%/yr)**")
        vol_cols = st.columns(3)
        vols_input: list[float] = []
        for i in range(6):
            with vol_cols[i % 3]:
                default_val = (
                    market.annual_volatilities[i] * 100
                    if use_current
                    else ref_market.annual_volatilities[i] * 100
                )
                val = st.number_input(
                    f"{mp_defaults.ALL_ASSET_NAMES[i]}",
                    value=default_val,
                    step=0.5,
                    key=f"vol_{i}",
                )
                vols_input.append(val)

        # Correlation preset
        st.markdown("**Correlation Matrix**")
        corr_preset = st.radio(
            "Correlation preset",
            ["Academic Consensus", "Crisis-Aware", "Custom"],
            index=0,
            horizontal=True,
        )
        if corr_preset == "Academic Consensus":
            correlation_matrix = [list(row) for row in mp_defaults.CORRELATION_CONSENSUS]
        elif corr_preset == "Crisis-Aware":
            correlation_matrix = [list(row) for row in mp_defaults.CORRELATION_CRISIS_AWARE]
        else:
            # Editable correlation matrix
            if use_current and len(market.correlation_matrix) == 6:
                corr_df = pd.DataFrame(
                    market.correlation_matrix,
                    index=mp_defaults.ALL_ASSET_NAMES,
                    columns=mp_defaults.ALL_ASSET_NAMES,
                )
            else:
                corr_df = pd.DataFrame(
                    mp_defaults.CORRELATION_CONSENSUS,
                    index=mp_defaults.ALL_ASSET_NAMES,
                    columns=mp_defaults.ALL_ASSET_NAMES,
                )
            edited_corr = st.data_editor(corr_df, key="corr_editor")
            correlation_matrix = edited_corr.values.tolist()

    else:
        # 2-asset US Only
        col1, col2 = st.columns(2)
        with col1:
            stock_return = st.number_input(
                "Expected Stock Return (%/yr)",
                value=market.expected_annual_returns[0] * 100 if current_assets_count == 2 else 8.0,
                step=0.5,
            )
            stock_vol = st.number_input(
                "Stock Volatility (%/yr)",
                value=market.annual_volatilities[0] * 100 if current_assets_count == 2 else 16.0,
                step=1.0,
            )
        with col2:
            bond_return = st.number_input(
                "Expected Bond Return (%/yr)",
                value=market.expected_annual_returns[1] * 100 if current_assets_count == 2 else 5.0,
                step=0.5,
            )
            bond_vol = st.number_input(
                "Bond Volatility (%/yr)",
                value=market.annual_volatilities[1] * 100 if current_assets_count == 2 else 7.0,
                step=0.5,
            )
        returns_input = [stock_return, bond_return]
        vols_input = [stock_vol, bond_vol]

        correlation = st.number_input(
            "Stock-Bond Correlation",
            min_value=-1.0,
            max_value=1.0,
            value=market.correlation_matrix[0][1] if current_assets_count == 2 else 0.0,
            step=0.1,
        )
        correlation_matrix = [
            [1.0, correlation],
            [correlation, 1.0],
        ]

    inflation_mean = st.number_input(
        "Long-Run Inflation (%/yr)",
        value=market.inflation_mean * 100,
        step=0.25,
    )

with st.expander("Investment Fees", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        expense_ratio = (
            st.number_input(
                "Fund Expense Ratio (%/yr)",
                value=market.expense_ratio * 100,
                step=0.05,
                min_value=0.0,
                max_value=5.0,
                help="Weighted average fund expense ratio (e.g. 0.10% for index funds)",
            )
            / 100
        )
    with col2:
        aum_fee = (
            st.number_input(
                "Platform/AUM Fee (%/yr)",
                value=market.aum_fee * 100,
                step=0.05,
                min_value=0.0,
                max_value=5.0,
                help="Brokerage or platform fee charged on assets",
            )
            / 100
        )
    with col3:
        advisory_fee = (
            st.number_input(
                "Advisory Fee (%/yr)",
                value=market.advisory_fee * 100,
                step=0.05,
                min_value=0.0,
                max_value=5.0,
                help="Financial advisor fee",
            )
            / 100
        )
    total_fee = expense_ratio + aum_fee + advisory_fee
    st.caption(f"Total annual fee drag: {total_fee:.2%}")

# --- Return Model ---
st.subheader("Return Model")
return_model_options = ["mvn", "student_t", "regime_switching"]
return_model_labels = {
    "mvn": "Multivariate Normal",
    "student_t": "Student-t (fat tails)",
    "regime_switching": "Regime Switching (Markov)",
}
return_model = st.selectbox(
    "Return Distribution",
    return_model_options,
    index=return_model_options.index(market.return_model)
    if market.return_model in return_model_options
    else 0,
    format_func=lambda x: return_model_labels.get(x, x),
)

degrees_of_freedom: float | None = market.degrees_of_freedom
if return_model == "student_t":
    degrees_of_freedom = st.number_input(
        "Degrees of Freedom",
        min_value=2.1,
        max_value=100.0,
        value=market.degrees_of_freedom or 5.0,
        step=0.5,
        help="Lower values = fatter tails. Typical range: 4-8.",
    )

regime_switching_config: RegimeSwitchingConfig | None = None
if return_model == "regime_switching":
    st.markdown("**Regime Parameters**")
    st.caption("Define 2-3 market regimes (bull/normal/bear) with transition probabilities.")

    n_regimes = st.radio("Number of Regimes", [2, 3], index=1, horizontal=True)

    # Regime defaults for 6-asset or 2-asset
    if n_assets == 6:
        regime_defaults = [
            {
                "name": "Bull",
                "returns": [12.0, 11.0, 14.0, 5.0, 4.5, 7.0],
                "vols": [12.0, 14.0, 19.0, 5.0, 7.0, 9.0],
                "infl": 2.5,
                "infl_vol": 0.8,
            },
            {
                "name": "Normal",
                "returns": [8.0, 7.5, 8.5, 5.0, 4.5, 6.0],
                "vols": [16.0, 18.0, 23.0, 7.0, 9.0, 12.0],
                "infl": 3.0,
                "infl_vol": 1.0,
            },
            {
                "name": "Bear",
                "returns": [-5.0, -8.0, -12.0, 1.0, 0.5, -2.0],
                "vols": [25.0, 28.0, 35.0, 8.0, 11.0, 15.0],
                "infl": 4.5,
                "infl_vol": 2.0,
            },
        ]
    else:
        regime_defaults = [
            {
                "name": "Bull",
                "returns": [12.0, 5.0],
                "vols": [12.0, 4.0],
                "infl": 2.5,
                "infl_vol": 0.8,
            },
            {
                "name": "Normal",
                "returns": [8.0, 5.0],
                "vols": [16.0, 7.0],
                "infl": 3.0,
                "infl_vol": 1.0,
            },
            {
                "name": "Bear",
                "returns": [-5.0, 1.0],
                "vols": [25.0, 8.0],
                "infl": 4.5,
                "infl_vol": 2.0,
            },
        ]

    regimes: list[mp_schema.RegimeConfig] = []
    for ri in range(n_regimes):
        d = regime_defaults[ri]
        with st.expander(f"Regime {ri + 1}: {d['name']}", expanded=ri == 0):
            r_name = st.text_input("Name", value=d["name"], key=f"regime_name_{ri}")

            # Returns and vols per asset
            r_returns: list[float] = []
            r_vols: list[float] = []
            if n_assets == 6:
                st.markdown("*Stocks*")
                sc1, sc2, sc3 = st.columns(3)
                for ai in range(3):
                    with [sc1, sc2, sc3][ai]:
                        rv = st.number_input(
                            f"{mp_defaults.STOCK_ASSET_NAMES[ai]} Ret (%)",
                            value=d["returns"][ai],
                            step=1.0,
                            key=f"r_ret_{ri}_{ai}",
                        )
                        r_returns.append(rv)
                        vv = st.number_input(
                            f"{mp_defaults.STOCK_ASSET_NAMES[ai]} Vol (%)",
                            value=d["vols"][ai],
                            step=1.0,
                            key=f"r_vol_{ri}_{ai}",
                        )
                        r_vols.append(vv)
                st.markdown("*Bonds*")
                bc1, bc2, bc3 = st.columns(3)
                for ai in range(3):
                    with [bc1, bc2, bc3][ai]:
                        rv = st.number_input(
                            f"{mp_defaults.BOND_ASSET_NAMES[ai]} Ret (%)",
                            value=d["returns"][ai + 3],
                            step=0.5,
                            key=f"r_ret_{ri}_{ai + 3}",
                        )
                        r_returns.append(rv)
                        vv = st.number_input(
                            f"{mp_defaults.BOND_ASSET_NAMES[ai]} Vol (%)",
                            value=d["vols"][ai + 3],
                            step=0.5,
                            key=f"r_vol_{ri}_{ai + 3}",
                        )
                        r_vols.append(vv)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    r_ret_s = st.number_input(
                        "Stock Return (%)", value=d["returns"][0], step=1.0, key=f"r_ret_s_{ri}"
                    )
                    r_vol_s = st.number_input(
                        "Stock Vol (%)", value=d["vols"][0], step=1.0, key=f"r_vol_s_{ri}"
                    )
                    r_returns.extend([r_ret_s])
                    r_vols.extend([r_vol_s])
                with c2:
                    r_ret_b = st.number_input(
                        "Bond Return (%)", value=d["returns"][1], step=0.5, key=f"r_ret_b_{ri}"
                    )
                    r_vol_b = st.number_input(
                        "Bond Vol (%)", value=d["vols"][1], step=0.5, key=f"r_vol_b_{ri}"
                    )
                    r_returns.extend([r_ret_b])
                    r_vols.extend([r_vol_b])

            r_infl = st.number_input(
                "Inflation (%)", value=d["infl"], step=0.25, key=f"r_infl_{ri}"
            )
            r_infl_vol = st.number_input(
                "Inflation Vol (%)", value=d["infl_vol"], step=0.25, key=f"r_infl_vol_{ri}"
            )

            # Use the same correlation matrix for regimes
            regime_corr = correlation_matrix

            regimes.append(
                mp_schema.RegimeConfig(
                    name=r_name,
                    expected_annual_returns=[r / 100 for r in r_returns],
                    annual_volatilities=[v / 100 for v in r_vols],
                    correlation_matrix=[list(row) for row in regime_corr],
                    inflation_mean=r_infl / 100,
                    inflation_vol=r_infl_vol / 100,
                )
            )

    st.markdown("**Transition Matrix** (monthly probability of switching)")
    trans_defaults_2 = [[0.97, 0.03], [0.05, 0.95]]
    trans_defaults_3 = [[0.95, 0.04, 0.01], [0.03, 0.94, 0.03], [0.02, 0.05, 0.93]]
    trans_defaults = trans_defaults_3 if n_regimes == 3 else trans_defaults_2

    transition_matrix: list[list[float]] = []
    for i in range(n_regimes):
        cols_tm = st.columns(n_regimes)
        row: list[float] = []
        for j in range(n_regimes):
            with cols_tm[j]:
                val = st.number_input(
                    f"P({regimes[i].name}\u2192{regimes[j].name})",
                    min_value=0.0,
                    max_value=1.0,
                    value=trans_defaults[i][j],
                    step=0.01,
                    key=f"trans_{i}_{j}",
                    format="%.2f",
                )
                row.append(val)
        transition_matrix.append(row)

    regime_switching_config = mp_schema.RegimeSwitchingConfig(
        regimes=regimes,
        transition_matrix=transition_matrix,
        initial_regime=1 if n_regimes == 3 else 0,
    )

# --- Stress Scenarios ---
st.subheader("Stress Scenarios")
st.caption("Deterministic overlays that modify simulation paths for specific market conditions")

stress_options = {
    "crash": "Market Crash (-38% over 12mo, V-shaped recovery)",
    "lost_decade": "Lost Decade (near-zero real returns)",
    "high_inflation": "High Inflation (6-8% annualized)",
    "sequence_risk": "Sequence Risk (poor returns early in retirement)",
}

existing_scenarios = {s.scenario_type: s for s in sim_config.stress_scenarios}
selected_scenarios: list[mp_schema.StressScenario] = []

for stype, label in stress_options.items():
    existing = existing_scenarios.get(stype)
    enabled = st.checkbox(label, value=existing is not None, key=f"stress_{stype}")
    if enabled:
        col1, col2, col3 = st.columns(3)
        with col1:
            s_age = st.number_input(
                "Start Age",
                min_value=18.0,
                max_value=120.0,
                value=existing.start_age if existing else 65.0,
                step=1.0,
                key=f"stress_age_{stype}",
            )
        with col2:
            s_dur = st.number_input(
                "Duration (months)",
                min_value=1,
                max_value=360,
                value=existing.duration_months if existing else (12 if stype == "crash" else 120),
                key=f"stress_dur_{stype}",
            )
        with col3:
            s_sev = st.number_input(
                "Severity",
                min_value=0.1,
                max_value=3.0,
                value=existing.severity if existing else 1.0,
                step=0.1,
                key=f"stress_sev_{stype}",
            )
        selected_scenarios.append(
            mp_schema.StressScenario(
                name=stype,
                scenario_type=stype,
                start_age=s_age,
                duration_months=int(s_dur),
                severity=s_sev,
            )
        )

# --- Simulation Settings ---
st.subheader("Simulation Settings")
sim_preset = st.radio(
    "Quality Preset",
    ["Fast (1,000)", "Balanced (5,000)", "Deep (20,000)"],
    index=1 if sim_config.preset is None else {"fast": 0, "balanced": 1, "deep": 2}[sim_config.preset],
    horizontal=True,
)
preset_map = {"Fast (1,000)": "fast", "Balanced (5,000)": "balanced", "Deep (20,000)": "deep"}
sim_preset_key = preset_map[sim_preset]

seed = st.number_input("Random Seed", value=sim_config.seed, min_value=0)

# --- Save ---
if st.button("Save Portfolio & Settings", type="primary"):
    try:
        # Build allocations
        def build_alloc(s: dict) -> mp_schema.AssetAllocation:
            p = s["preset"]
            s_pct = s["stock_pct"]
            if p in ("Global Diversified", "Custom"):
                s_reg = [s["stock_us"] / 100, s["stock_exus"] / 100, s["stock_em"] / 100]
                b_reg = [s["bond_us"] / 100, s["bond_exus"] / 100, s["bond_em"] / 100]
                wts = mp_defaults.build_global_weights(
                    stock_pct=s_pct / 100,
                    stock_regional=s_reg,
                    bond_regional=b_reg,
                )
                assets = [mp_schema.AssetClass(name=name, weight=w) for name, w in zip(mp_defaults.ALL_ASSET_NAMES, wts)]
            else:
                assets = [
                    mp_schema.AssetClass(name="US Stocks", weight=s_pct / 100),
                    mp_schema.AssetClass(name="US Bonds", weight=(100 - s_pct) / 100),
                ]
            
            gl_path = None
            if s["use_glide"]:
                if len(assets) == 6:
                    end_wts = mp_defaults.build_global_weights(
                        stock_pct=s["gp_end_stock"] / 100,
                        stock_regional=s_reg if p in ("Global Diversified", "Custom") else None,
                        bond_regional=b_reg if p in ("Global Diversified", "Custom") else None,
                    )
                else:
                    end_wts = [s["gp_end_stock"] / 100, (100 - s["gp_end_stock"]) / 100]
                gl_path = mp_schema.GlidePath(
                    start_age=int(s["gp_start_age"]),
                    start_weights=[a.weight for a in assets],
                    end_age=int(s["gp_end_age"]),
                    end_weights=end_wts,
                )
            return mp_schema.AssetAllocation(assets=assets, glide_path=gl_path)

        if use_all_accounts:
            allocs = [build_alloc(st.session_state["alloc_settings"]["taxable"])]
        else:
            # We must output exactly len(plan_data.accounts) asset allocations
            allocs = []
            for account in plan_data.accounts:
                acct_type = account.account_type.lower()
                s = st.session_state["alloc_settings"].get(acct_type, st.session_state["alloc_settings"]["taxable"])
                allocs.append(build_alloc(s))

        # Check that all allocations have the same number of assets
        num_assets_list = [len(alloc.assets) for alloc in allocs]
        if len(set(num_assets_list)) > 1:
            st.error("All accounts must use the same number of assets (either all 6-asset or all 2-asset configurations).")
            st.stop()

        new_market = mp_schema.MarketAssumptions(
            asset_allocations=allocs,
            expected_annual_returns=[r / 100 for r in returns_input],
            annual_volatilities=[v / 100 for v in vols_input],
            correlation_matrix=[list(row) for row in correlation_matrix],
            inflation_mean=inflation_mean / 100,
            inflation_vol=market.inflation_vol,
            expense_ratio=expense_ratio,
            aum_fee=aum_fee,
            advisory_fee=advisory_fee,
            return_model=return_model,
            degrees_of_freedom=degrees_of_freedom if return_model == "student_t" else None,
            regime_switching=regime_switching_config
            if return_model == "regime_switching"
            else None,
        )
        new_sim = mp_schema.SimulationConfig(
            seed=int(seed),
            preset=sim_preset_key,
            stress_scenarios=selected_scenarios,
        )

        st.session_state["market"] = new_market
        st.session_state["sim_config"] = new_sim
        st.success("Portfolio & settings saved!")
    except Exception as e:
        st.error(f"Validation error: {e}")
