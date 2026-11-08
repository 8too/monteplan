# Model Limitations

monteplan is an educational and planning tool. Understanding its limitations is essential to interpreting results responsibly.

## General Limitations

**This is not financial advice.** Results are simulations based on simplified models and assumptions. Always consult a qualified financial advisor for real planning decisions.

### Simplified Tax Model

- Only US federal taxes are modeled. State and local taxes are not included.
- The model assumes a single filing status for the entire simulation horizon.
- Tax-loss harvesting, estate taxes, and alternative minimum tax (AMT) are not modeled.
- Capital gains are simplified -- the model does not track individual lot-level cost basis.
- Tax bracket tables are the 2026 values, indexed by simulated inflation in later years; they do not adjust for future legislation.

### No Behavioral Dynamics

- The model assumes perfect adherence to the chosen spending policy (no panic selling, no lifestyle inflation beyond the plan).
- Contribution rates are fixed; real-world saving behavior varies with life events.
- Healthcare costs, long-term care, and other age-dependent expenses are not explicitly modeled.

### Return Model Assumptions

- **MVN:** Returns are independently and identically distributed (i.i.d.) across time. Real markets exhibit momentum, mean reversion, and volatility clustering.
- **Student-t:** Captures fat tails but not skewness or time-varying volatility.
- **Bootstrap:** Limited to the range of observed historical data. Cannot generate worse-than-observed crashes. Sensitive to the choice of historical sample period.
- **Regime switching:** Regime definitions are user-specified, not estimated from data. The number of regimes and transition probabilities are assumptions, not inferences.

### Inflation Simplifications

- Inflation is modeled as a single aggregate rate, not component-based (healthcare inflation often outpaces general CPI).
- The OU process assumes inflation is stationary and mean-reverting. Structural shifts (like the 1970s) require regime switching.

### Portfolio Simplifications

- The model assumes continuous rebalancing is possible at zero cost (no transaction costs, bid-ask spreads, or market impact).
- Asset classes are represented as broad categories; individual security selection is not modeled.
- Real estate, private equity, and other illiquid assets cannot be properly modeled in a mark-to-market framework.

### Social Security / Guaranteed Income

- Benefit amounts are assumed to be known and fixed (in real dollars before COLA).
- The model does not account for Social Security claiming strategy optimization (when to claim).
- Pension and annuity default risk is not modeled.

### Longevity Risk

- The simulation runs to a fixed `end_age`. There is no stochastic mortality model.
- Users must choose a conservative `end_age` to account for the possibility of living longer than expected.

## What This Means for Interpretation

1. **Success probability is approximate.** A result of "80% success" does not mean there is exactly an 80% chance of a plan working. It means that under the assumed model, 80% of simulated paths survived.

2. **Sensitivity analysis is your friend.** Because all results depend on assumptions, test how sensitive your plan is to changes in key parameters. A plan that is robust across a range of assumptions is more reliable than one that depends on specific values.

3. **Use multiple return models.** Run the same plan under MVN, Student-t, and regime switching. If the plan succeeds under all three, it is more robust.

4. **Stress test explicitly.** Overlay crash and high-inflation scenarios to see how the plan handles adverse conditions.

5. **Update regularly.** Re-run simulations as your financial situation changes, and update market assumptions to reflect current conditions.
