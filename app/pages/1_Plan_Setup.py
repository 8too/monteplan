"""Plan Setup page — ages, income, spending, accounts, events."""

from __future__ import annotations

import streamlit as st

from monteplan.config.defaults import default_plan
from monteplan.config.schema import AccountConfig, DiscreteEvent, GuaranteedIncomeStream, PlanConfig

st.set_page_config(page_title="Plan Setup — MontePlan", layout="wide")
st.title("Plan Setup")

# Initialize defaults if needed
if "plan" not in st.session_state:
    st.session_state["plan"] = default_plan()

plan: PlanConfig = st.session_state["plan"]

st.subheader("Ages")
col1, col2, col3 = st.columns(3)
with col1:
    current_age = st.number_input("Current Age", min_value=18, max_value=100, value=plan.current_age)
with col2:
    retirement_age = st.number_input(
        "Retirement Age", min_value=1, max_value=100, value=max(plan.retirement_age, current_age + 1)
    )
with col3:
    end_age = st.number_input(
        "Plan End Age", min_value=retirement_age + 1, max_value=120, value=max(plan.end_age, retirement_age + 1)
    )

st.subheader("Income & Spending")
col1, col2, col3 = st.columns(3)
with col1:
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0.0, value=plan.monthly_income, step=500.0
    )
with col2:
    monthly_spending = st.number_input(
        "Monthly Spending ($)", min_value=0.0, value=plan.monthly_spending, step=500.0
    )
with col3:
    income_growth_rate = st.number_input(
        "Annual Income Growth (%)",
        min_value=-10.0,
        max_value=20.0,
        value=plan.income_growth_rate * 100,
        step=0.5,
        help="Annual real income growth rate (e.g. 2% for raises above inflation)",
    )

income_end_age = st.number_input(
    "Income End Age",
    min_value=1,
    max_value=end_age,
    value=min(plan.income_end_age or retirement_age, end_age),
)

st.subheader("Accounts")
n_accounts = st.number_input(
    "Number of Accounts", min_value=1, max_value=10, value=len(plan.accounts)
)

accounts: list[AccountConfig] = []
for i in range(int(n_accounts)):
    existing = plan.accounts[i] if i < len(plan.accounts) else AccountConfig(
        account_type="taxable", balance=0, annual_contribution=0
    )
    with st.expander(f"Account {i + 1}: {existing.account_type.title()}", expanded=i < 3):
        from app.components.forms import account_form
        acct = account_form(i, existing)
        accounts.append(acct)

# Guaranteed Income Streams
st.subheader("Guaranteed Income")
st.caption("Recurring income streams like Social Security, pensions, and annuities")

n_gi = st.number_input(
    "Number of Income Streams", min_value=0, max_value=10,
    value=len(plan.guaranteed_income), key="n_gi",
)

gi_streams: list[GuaranteedIncomeStream] = []
for i in range(int(n_gi)):
    existing_gi = plan.guaranteed_income[i] if i < len(plan.guaranteed_income) else GuaranteedIncomeStream(
        name="Social Security", monthly_amount=2000, start_age=67,
    )
    with st.expander(f"Income {i + 1}: {existing_gi.name}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gi_name = st.text_input("Name", value=existing_gi.name, key=f"gi_name_{i}")
            gi_amount = st.number_input(
                "Monthly Amount ($)", value=existing_gi.monthly_amount,
                step=100.0, min_value=0.0, key=f"gi_amount_{i}",
            )
        with col2:
            gi_start = st.number_input(
                "Start Age", value=existing_gi.start_age,
                min_value=18.0, max_value=120.0, step=1.0, key=f"gi_start_{i}",
            )
            gi_cola = st.number_input(
                "COLA (%/yr)", value=existing_gi.cola_rate * 100,
                min_value=0.0, max_value=10.0, step=0.5, key=f"gi_cola_{i}",
                help="Annual cost-of-living adjustment (e.g. 2% for SS)",
            ) / 100
        gi_has_end = st.checkbox(
            "Has end age", value=existing_gi.end_age is not None, key=f"gi_has_end_{i}",
        )
        gi_end: float | None = None
        if gi_has_end:
            gi_end = st.number_input(
                "End Age", value=existing_gi.end_age or 90.0,
                min_value=gi_start, max_value=120.0, step=1.0, key=f"gi_end_{i}",
            )
        gi_streams.append(GuaranteedIncomeStream(
            name=gi_name, monthly_amount=gi_amount, start_age=gi_start,
            cola_rate=gi_cola, end_age=gi_end,
        ))

# Discrete Events
st.subheader("Discrete Events")
st.caption("One-time financial events (e.g. home purchase, inheritance, Social Security start)")

n_events = st.number_input(
    "Number of Events", min_value=0, max_value=20, value=len(plan.discrete_events), key="n_events"
)

events: list[DiscreteEvent] = []
for i in range(int(n_events)):
    existing_ev = plan.discrete_events[i] if i < len(plan.discrete_events) else DiscreteEvent(
        age=65.0, amount=0.0, description=""
    )
    with st.expander(f"Event {i + 1}: {existing_ev.description or 'New Event'}", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            ev_age = st.number_input(
                "Age", min_value=18.0, max_value=120.0,
                value=existing_ev.age, step=0.5, key=f"ev_age_{i}",
            )
        with col2:
            ev_amount = st.number_input(
                "Amount ($)",
                value=existing_ev.amount, step=1000.0, key=f"ev_amount_{i}",
                help="Positive = inflow, Negative = outflow",
            )
        with col3:
            ev_desc = st.text_input(
                "Description", value=existing_ev.description, key=f"ev_desc_{i}",
            )
        events.append(DiscreteEvent(age=ev_age, amount=ev_amount, description=ev_desc))

if st.button("Save Plan", type="primary"):
    try:
        new_plan = PlanConfig(
            current_age=int(current_age),
            retirement_age=int(retirement_age),
            end_age=int(end_age),
            accounts=accounts,
            monthly_income=monthly_income,
            monthly_spending=monthly_spending,
            income_end_age=int(income_end_age),
            income_growth_rate=income_growth_rate / 100,
            discrete_events=events,
            guaranteed_income=gi_streams,
        )
        st.session_state["plan"] = new_plan
        st.success("Plan saved!")
    except Exception as e:
        st.error(f"Validation error: {e}")
