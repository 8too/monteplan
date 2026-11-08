# Tax Model Mathematics

## US Federal Tax Model

### Ordinary Income Tax

Ordinary income (traditional account withdrawals, earned income) is taxed through progressive brackets after applying the standard deduction:

$$\text{taxable income} = \text{gross income} - \text{standard deduction}$$

Tax is computed by applying marginal rates to each bracket:

$$\text{tax} = \sum_{k=1}^{K} r_k \cdot \min(\max(\text{taxable income} - b_{k-1}, 0),\, b_k - b_{k-1})$$

where $r_k$ is the marginal rate for bracket $k$ and $[b_{k-1}, b_k)$ is the bracket range.

### 2026 Federal Tax Brackets (Single)

Source: IRS Revenue Procedure 2025-32.

| Taxable Income | Rate |
|---|---|
| $0 -- $12,400 | 10% |
| $12,401 -- $50,400 | 12% |
| $50,401 -- $105,700 | 22% |
| $105,701 -- $201,775 | 24% |
| $201,776 -- $256,225 | 32% |
| $256,226 -- $640,600 | 35% |
| Over $640,600 | 37% |

Standard deduction (2026): $16,100 (single), $32,200 (married filing jointly).

### Long-Term Capital Gains

Net long-term gains are taxed at separate rates. Short-term gains count as ordinary income.

| Taxable Income (Single) | LTCG Rate |
|---|---|
| Up to $49,450 | 0% |
| $49,451 -- $545,500 | 15% |
| Over $545,500 | 20% |

The gains are the top layer of taxable income: ordinary income fills the brackets first, and
each gains rate applies to the part of the gains that falls into its bracket. Any standard
deduction left unused by ordinary income reduces the gains first. With ordinary taxable income
$O$ and taxable gains $G$, the gains tax is

$$\text{LTCG tax} = T_\text{LTCG}(O + G) - T_\text{LTCG}(O)$$

where $T_\text{LTCG}$ applies the gains rates progressively to taxable income.

### Inflation Indexing

The IRS indexes the standard deduction and all bracket thresholds for inflation each year.
The engine takes the tax tables to be those of the simulation's first year and multiplies
every amount in a later year by that path's cumulative simulated inflation up to the start of
the year. Income that keeps pace with inflation thus pays the same real tax. The NIIT
thresholds ($200,000 single, $250,000 married filing jointly) are fixed by statute and not
indexed.

### Vectorized Implementation

Tax calculations are vectorized across all simulation paths. The engine computes taxes for all paths simultaneously using numpy operations rather than per-path Python loops:

```python
compute_annual_tax_vectorized(
    ordinary_income: NDArray,    # (n_paths,)
    ltcg: NDArray,               # (n_paths,)
    filing_status: str,
    inflation_index: NDArray,    # (n_paths,) price level relative to the table year
) -> NDArray                     # (n_paths,)
```

## Required Minimum Distributions (RMDs)

Traditional account holders must take minimum distributions starting at age 73 (per current IRS rules). The RMD is calculated as:

$$\text{RMD} = \frac{\text{prior year balance}}{\text{divisor}(\text{age})}$$

### IRS Uniform Lifetime Table (Selected Values)

| Age | Divisor | Implied Rate |
|---|---|---|
| 73 | 26.5 | 3.8% |
| 75 | 24.6 | 4.1% |
| 80 | 20.2 | 5.0% |
| 85 | 16.0 | 6.3% |
| 90 | 12.2 | 8.2% |
| 95 | 8.6 | 11.6% |
| 100 | 6.4 | 15.6% |

The divisor table is loaded from `taxes/tables/rmd_divisors.yaml`.

## Flat Tax Model

The flat tax model applies a single effective rate to all taxable income:

$$\text{tax} = \text{rate} \times \text{gross income}$$

This is simpler but less realistic. It's useful for:

- Quick analysis where tax precision isn't critical
- Non-US users who want to model a different tax system
- Sensitivity analysis to isolate tax effects

## Withdrawal Tax Treatment

| Account Type | Tax Treatment |
|---|---|
| Taxable | Capital gains tax on appreciation only |
| Traditional (401k/IRA) | Ordinary income tax on full withdrawal |
| Roth | Tax-free |

### Tax Gross-Up

When withdrawing from a traditional account to meet spending needs, the engine applies a tax gross-up:

$$\text{gross withdrawal} = \frac{\text{needed spending}}{1 - \text{effective rate}}$$

This ensures the after-tax amount meets the spending target.
