# Scaling Plan

The `10m` sequence idea is a reasonable scale target, but it is not a magic threshold. What matters is:

- how many unique input windows we have,
- how correlated those windows are,
- whether the targets are leakage-safe,
- and whether the extra scale is actually relevant to SPY.

## Honest Multipliers

These really do multiply the training problem:

1. More assets
2. More intraday timeframes
3. Sliding windows with a chosen stride
4. More years of history
5. More targets, if we train separate heads or separate examples

This one does **not** multiply sequence count:

6. Moving averages

Moving averages are valuable features, but they increase feature richness and compute cost, not the number of unique windows.

## What This Means For Us

Daily-only data is nowhere near enough. Even a broad daily panel gets us on the order of hundreds of thousands of unique windows, not tens of millions.

The clean path to `10m+` labeled examples is:

- a multi-asset panel,
- at least one high-frequency clock such as `1min` or `5min`,
- a non-trivial history window,
- multiple prediction targets,
- and a stride that controls redundant overlap.

## Recommended Build Order

1. Keep the current daily panel as a low-cost sanity set.
2. Add a scaling planner so we can count expected windows before building new data pipelines.
3. Build a `core intraday ETF + macro` dataset first.
4. Only after that, widen to futures and a stock universe.
5. Use moving averages across `5min`, `15min`, `1h`, `daily`, `weekly`, and `monthly` as features, not as fake data multipliers.

## Real Blocker

The real blocker is no longer the GPU. It is the data source.

The current repo uses a daily `yfinance`-style setup. That is fine for the first research phase, but it is not the right backend for multi-year, multi-asset, high-frequency research at the scale we are talking about.
