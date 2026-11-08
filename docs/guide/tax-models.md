# Tax Models

monteplan models taxes on withdrawals during retirement. Two tax models are available:

## Flat Tax

A simple flat effective rate applied to all taxable income. Good for quick analysis or non-US users.

```python
policies = PolicyBundle(
    tax_model="flat",
    tax_rate=0.22,  # 22% flat effective rate
)
```

## US Federal Tax

Progressive tax brackets with separate rates for ordinary income and long-term capital gains (LTCG). Bracket tables are loaded from YAML data files.

```python
policies = PolicyBundle(
    tax_model="us_federal",
    filing_status="single",         # or "married_jointly"
)
```

### How It Works

1. **Traditional account withdrawals** are taxed as ordinary income through progressive brackets
2. **Taxable account gains** are taxed at LTCG rates (0%, 15%, or 20% depending on income), as the top layer of taxable income
3. **Roth withdrawals** are tax-free
4. **Standard deduction** is applied before bracket calculation
5. **Inflation indexing**: the 2026 standard deduction and bracket thresholds are indexed by each path's simulated inflation in later years

### Filing Status

| Status | Standard Deduction (2026) | Bracket Widths |
|---|---|---|
| `single` | $16,100 | Standard |
| `married_jointly` | $32,200 | ~2x single |

### Required Minimum Distributions (RMDs)

When `tax_model="us_federal"`, the engine enforces IRS Required Minimum Distributions from traditional accounts starting at age 73. RMDs are calculated using the IRS Uniform Lifetime Table.

The RMD for a year is:

```
RMD = prior_year_balance / divisor(age)
```

The divisor decreases with age (e.g., 26.5 at age 73, 20.2 at age 80, 12.2 at age 90).

### Withdrawal Ordering

The `withdrawal_order` field controls which accounts are drawn from first:

```python
policies = PolicyBundle(
    withdrawal_order=["taxable", "traditional", "roth"],  # default
)
```

The default order (taxable first, Roth last) is generally tax-efficient because:

- Taxable withdrawals may have low or zero tax (return of basis)
- Traditional withdrawals are taxed as ordinary income
- Roth withdrawals are always tax-free (best saved for last)

### Vectorized Implementation

Tax calculations are fully vectorized across all paths using `compute_annual_tax_vectorized()`. There are no per-path Python loops in the hot path.

See [Tax Math](../math/taxes.md) for bracket tables and calculation details.
