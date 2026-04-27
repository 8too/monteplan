"""Portfolio page — asset allocation, market assumptions, return model, stress scenarios."""

from __future__ import annotations

import streamlit as st

from monteplan.config.defaults import (
    ALL_ASSET_NAMES,
    BOND_ASSET_NAMES,
    CORRELATION_CONSENSUS,
    CORRELATION_CRISIS_AWARE,
    STOCK_ASSET_NAMES,
    build_global_weights,
    default_market,
    default_sim_config,
)
from monteplan.config.schema import (
    AssetClass,
    GlidePath,
    MarketAssumptions,
    RegimeConfig,
    RegimeSwitchingConfig,
    SimulationConfig,
    StressScenario,
)

st.set_page_config(page_title="Portfolio — MontePlan", layout="wide")

from app.components.theme import register_theme

register_theme()

st.title("Portfolio & Assumptions")

# Initialize defaults
if "market" not in st.session_state:
    st.session_state["market"] = default_market()
if "sim_config" not in st.session_state:
    st.session_state["sim_config"] = default_sim_config()

market: MarketAssumptions = st.session_state["market"]
sim_config: SimulationConfig = st.session_state["sim_config"]

# --- Portfolio Preset ---
st.subheader("Asset Allocation")

preset = st.radio(
    "Portfolio Preset",
    ["Global Diversified", "US Only", "Custom"],
    index=0,
    horizontal=True,
    help="Global: 6-asset US/Ex-US/EM. US Only: 2-asset US Stocks + US Bonds.",
)

is_global = preset == "Global Diversified"
is_custom = preset == "Custom"


def _current_stock_pct() -> int:
    """Infer current stock % from market state."""
    return int(round(sum(a.weight for a in market.assets if "Stock" in a.name) * 100))


def _current_regional(group: str) -> list[int]:
    """Infer current regional split [US, ExUS, EM] for 'stock' or 'bond' group."""
    keyword = "Stock" if group == "stock" else "Bond"
    weights = [a.weight for a in market.assets if keyword in a.name]
    if len(weights) != 3:
        return [60, 30, 10]
    total = sum(weights)
    if total < 1e-9:
        return [60, 30, 10]
    return [int(round(w / total * 100)) for w in weights]


# --- Stock / Bond Split ---
stock_pct = st.slider(
    "Stocks (%)",
    min_value=0,
    max_value=100,
    value=_current_stock_pct(),
    step=5,
)
bond_pct = 100 - stock_pct
st.write(f"Bonds: {bond_pct}%")

# --- Regional Splits (only for Global / Custom) ---
if is_global or is_custom:
    st.markdown("**Regional Split (Stocks)**")
    scol1, scol2, scol3 = st.columns(3)
    cur_stock_reg = _current_regional("stock")
    with scol1:
        stock_us = st.number_input(
            "US (%)",
            min_value=0,
            max_value=100,
            value=cur_stock_reg[0],
            step=5,
            key="stock_us",
        )
    with scol2:
        stock_exus = st.number_input(
            "Ex-US Dev (%)",
            min_value=0,
            max_value=100,
            value=cur_stock_reg[1],
            step=5,
            key="stock_exus",
        )
    with scol3:
        stock_em = st.number_input(
            "EM (%)",
            min_value=0,
            max_value=100,
            value=cur_stock_reg[2],
            step=5,
            key="stock_em",
        )
    stock_reg_sum = stock_us + stock_exus + stock_em
    if stock_reg_sum != 100:
        st.warning(f"Stock regional split sums to {stock_reg_sum}%, must be 100%.")

    same_bond_split = st.checkbox("Same regional split for bonds", value=True)
    if same_bond_split:
        bond_us, bond_exus, bond_em = stock_us, stock_exus, stock_em
    else:
        st.markdown("**Regional Split (Bonds)**")
        bcol1, bcol2, bcol3 = st.columns(3)
        cur_bond_reg = _current_regional("bond")
        with bcol1:
            bond_us = st.number_input(
                "US (%)",
                min_value=0,
                max_value=100,
                value=cur_bond_reg[0],
                step=5,
                key="bond_us",
            )
        with bcol2:
            bond_exus = st.number_input(
                "Ex-US Dev (%)",
                min_value=0,
                max_value=100,
                value=cur_bond_reg[1],
                step=5,
                key="bond_exus",
            )
        with bcol3:
            bond_em = st.number_input(
                "EM (%)",
                min_value=0,
                max_value=100,
                value=cur_bond_reg[2],
                step=5,
                key="bond_em",
            )
        bond_reg_sum = bond_us + bond_exus + bond_em
        if bond_reg_sum != 100:
            st.warning(f"Bond regional split sums to {bond_reg_sum}%, must be 100%.")

    stock_regional = [stock_us / 100, stock_exus / 100, stock_em / 100]
    bond_regional = [bond_us / 100, bond_exus / 100, bond_em / 100]

    # --- Bond Type ---
    bond_type = st.radio(
        "Bond Type",
        ["Aggregate (2% real)", "Treasuries Only (1.5% real)"],
        index=0,
        horizontal=True,
    )
    bond_type_key = "treasuries" if "Treasuries" in bond_type else "aggregate"

    # --- Computed Weights Display ---
    computed_weights = build_global_weights(
        stock_pct=stock_pct / 100,
        stock_regional=stock_regional,
        bond_regional=bond_regional,
    )
    st.markdown("**Computed Weights**")
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        for i, name in enumerate(STOCK_ASSET_NAMES):
            st.write(f"{name}: {computed_weights[i]:.0%}")
    with wcol2:
        for i, name in enumerate(BOND_ASSET_NAMES):
            st.write(f"{name}: {computed_weights[i + 3]:.0%}")

    n_assets = 6
    asset_names = list(ALL_ASSET_NAMES)
    asset_weights = computed_weights

else:
    # US Only — simple 2-asset
    bond_type_key = "aggregate"
    n_assets = 2
    asset_names = ["US Stocks", "US Bonds"]
    asset_weights = [stock_pct / 100, bond_pct / 100]

# --- Glide Path ---
use_glide = st.checkbox(
    "Enable Glide Path (age-based allocation shift)",
    value=market.glide_path is not None,
)
gp_start_age = 30
gp_end_age = 70
gp_end_stock = 40
if use_glide:
    col1, col2, col3 = st.columns(3)
    with col1:
        gp_start_age = st.number_input(
            "Glide Start Age",
            min_value=18,
            max_value=120,
            value=market.glide_path.start_age if market.glide_path else 30,
        )
    with col2:
        gp_end_age = st.number_input(
            "Glide End Age",
            min_value=gp_start_age + 1,
            max_value=120,
            value=market.glide_path.end_age if market.glide_path else 70,
        )
    with col3:
        gp_end_stock = st.slider(
            "End Stock Allocation (%)",
            min_value=0,
            max_value=100,
            step=5,
            value=int(
                sum(
                    market.glide_path.end_weights[i]
                    for i in range(len(market.glide_path.end_weights))
                    if i < len(market.assets) and "Stock" in market.assets[i].name
                )
                * 100
            )
            if market.glide_path
            else 40,
        )
    st.caption(
        f"Allocation shifts from {stock_pct}% stocks at age {gp_start_age} "
        f"to {gp_end_stock}% stocks at age {gp_end_age}"
    )

# --- Allocation area chart preview ---
from app.components.charts import allocation_area_chart

plan_data = st.session_state.get("plan")
_preview_current_age = plan_data.current_age if plan_data else 30
_preview_end_age = plan_data.end_age if plan_data else 90
_preview_assets = [{"name": asset_names[i], "weight": asset_weights[i]} for i in range(n_assets)]
_preview_gp = None
if use_glide:
    if n_assets == 6:
        # Build end weights preserving regional ratios
        end_weights = build_global_weights(
            stock_pct=gp_end_stock / 100,
            stock_regional=stock_regional if (is_global or is_custom) else None,
            bond_regional=bond_regional if (is_global or is_custom) else None,
        )
    else:
        end_weights = [gp_end_stock / 100, (100 - gp_end_stock) / 100]
    _preview_gp = {
        "start_age": gp_start_age,
        "start_weights": list(asset_weights),
        "end_age": gp_end_age,
        "end_weights": end_weights,
    }
_alloc_fig = allocation_area_chart(
    _preview_assets,
    _preview_gp,
    _preview_current_age,
    _preview_end_age,
)
st.plotly_chart(_alloc_fig, use_container_width=True)

# --- Market Assumptions ---
with st.expander("Market Assumptions", expanded=False):
    if n_assets == 6:
        # Get reference market for default return/vol values
        ref_market = default_market(bond_type=bond_type_key)

        # Use current market values if asset count matches, otherwise use reference
        use_current = len(market.assets) == 6

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
                    f"{ALL_ASSET_NAMES[i]}",
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
                    f"{ALL_ASSET_NAMES[i]}",
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
            correlation_matrix = [list(row) for row in CORRELATION_CONSENSUS]
        elif corr_preset == "Crisis-Aware":
            correlation_matrix = [list(row) for row in CORRELATION_CRISIS_AWARE]
        else:
            import pandas as pd

            # Editable correlation matrix
            if use_current and len(market.correlation_matrix) == 6:
                corr_df = pd.DataFrame(
                    market.correlation_matrix,
                    index=ALL_ASSET_NAMES,
                    columns=ALL_ASSET_NAMES,
                )
            else:
                corr_df = pd.DataFrame(
                    CORRELATION_CONSENSUS,
                    index=ALL_ASSET_NAMES,
                    columns=ALL_ASSET_NAMES,
                )
            edited_corr = st.data_editor(corr_df, key="corr_editor")
            correlation_matrix = edited_corr.values.tolist()

    else:
        # 2-asset US Only
        col1, col2 = st.columns(2)
        with col1:
            stock_return = st.number_input(
                "Expected Stock Return (%/yr)",
                value=market.expected_annual_returns[0] * 100 if len(market.assets) == 2 else 8.0,
                step=0.5,
            )
            stock_vol = st.number_input(
                "Stock Volatility (%/yr)",
                value=market.annual_volatilities[0] * 100 if len(market.assets) == 2 else 16.0,
                step=1.0,
            )
        with col2:
            bond_return = st.number_input(
                "Expected Bond Return (%/yr)",
                value=market.expected_annual_returns[1] * 100 if len(market.assets) == 2 else 5.0,
                step=0.5,
            )
            bond_vol = st.number_input(
                "Bond Volatility (%/yr)",
                value=market.annual_volatilities[1] * 100 if len(market.assets) == 2 else 7.0,
                step=0.5,
            )
        returns_input = [stock_return, bond_return]
        vols_input = [stock_vol, bond_vol]

        correlation = st.number_input(
            "Stock-Bond Correlation",
            min_value=-1.0,
            max_value=1.0,
            value=market.correlation_matrix[0][1] if len(market.assets) == 2 else 0.0,
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

    regimes: list[RegimeConfig] = []
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
                            f"{STOCK_ASSET_NAMES[ai]} Ret (%)",
                            value=d["returns"][ai],
                            step=1.0,
                            key=f"r_ret_{ri}_{ai}",
                        )
                        r_returns.append(rv)
                        vv = st.number_input(
                            f"{STOCK_ASSET_NAMES[ai]} Vol (%)",
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
                            f"{BOND_ASSET_NAMES[ai]} Ret (%)",
                            value=d["returns"][ai + 3],
                            step=0.5,
                            key=f"r_ret_{ri}_{ai + 3}",
                        )
                        r_returns.append(rv)
                        vv = st.number_input(
                            f"{BOND_ASSET_NAMES[ai]} Vol (%)",
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
                RegimeConfig(
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

    regime_switching_config = RegimeSwitchingConfig(
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
selected_scenarios: list[StressScenario] = []

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
            StressScenario(
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
        # Build asset list
        assets = [
            AssetClass(name=asset_names[i], weight=asset_weights[i]) for i in range(n_assets)
        ]

        # Build glide path weights
        glide_path: GlidePath | None = None
        if use_glide:
            if n_assets == 6:
                end_wts = build_global_weights(
                    stock_pct=gp_end_stock / 100,
                    stock_regional=stock_regional if (is_global or is_custom) else None,
                    bond_regional=bond_regional if (is_global or is_custom) else None,
                )
            else:
                end_wts = [gp_end_stock / 100, (100 - gp_end_stock) / 100]
            glide_path = GlidePath(
                start_age=int(gp_start_age),
                start_weights=list(asset_weights),
                end_age=int(gp_end_age),
                end_weights=end_wts,
            )

        new_market = MarketAssumptions(
            assets=assets,
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
            glide_path=glide_path,
        )
        new_sim = SimulationConfig(
            seed=int(seed),
            preset=sim_preset_key,
            stress_scenarios=selected_scenarios,
        )

        st.session_state["market"] = new_market
        st.session_state["sim_config"] = new_sim
        st.success("Portfolio & settings saved!")
    except Exception as e:
        st.error(f"Validation error: {e}")
