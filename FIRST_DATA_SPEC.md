# First Data Spec - SPY Direction And Signal Strength

This file defines the first training dataset for the project.

The goal is not to capture every possible source of alpha. The goal is to create a small, honest, leakage-safe dataset that gives a model enough market context to predict short-horizon SPY behavior better than a naive baseline.

## Recommended First Problem

Predict the next `5` trading day SPY move using information known at the close of day `t`.

Also create a secondary benchmark target for the next `1` trading day SPY move.

The model should output:

- the most likely move bucket,
- the full probability distribution across buckets, normalized to sum to `1`,
- and a derived signal-strength measure based on calibrated probabilities.

## Why This Dataset Shape

Do not train the first model on SPY price alone.

For a first project, SPY-only daily data is too small and too noisy. If we want a useful model, it needs market-state context:

- sector leadership,
- risk-on vs risk-off behavior,
- rates and credit pressure,
- volatility regime,
- broad trend and mean-reversion context,
- and a small number of simple breadth proxies.

## Phase 1 Data Scope

Use daily adjusted OHLCV data from Yahoo Finance.

Start with ETFs and market proxies only. Do not start with all S&P 500 constituents yet.

Reason:

- it is much easier to clean,
- it avoids early survivorship-bias headaches,
- it gives enough signal context for a first pass,
- and it lets us focus on getting the research pipeline right.

## Required Raw Series

### Core target instrument

- `SPY`

### Sector context

- `XLB`
- `XLE`
- `XLF`
- `XLI`
- `XLK`
- `XLP`
- `XLU`
- `XLV`
- `XLY`

### Equity style and market structure context

- `QQQ`
- `IWM`
- `DIA`
- `MDY`
- `RSP`
- `SMH`

### Rates, credit, and defensive context

- `TLT`
- `IEF`
- `SHY`
- `LQD`
- `HYG`

### Commodity / macro proxy context

- `GLD`
- `DBC`
- `UUP`

### Volatility context

- `^VIX`

This is already enough for a strong first pass.

Optional later additions:

- `XLC`
- `XLRE`

These are useful, but they have shorter live histories than the older sector ETFs, so keep them out of the first long-history dataset if we want to start around `2006`.

## Raw Fields To Keep

For each ticker, store:

- `date`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

Prefer building returns and indicators from `adj_close`.

Keep raw OHLCV because some features may later use intrabar range or volume behavior.

## First-Pass Derived Feature Blocks

Build features for each ticker and each ratio only using past data up to day `t`.

### 1. Return features

For each series:

- `ret_1d`
- `ret_3d`
- `ret_5d`
- `ret_10d`
- `ret_20d`
- `ret_60d`
- `ret_120d`
- `ret_252d`

Use log returns or percentage returns consistently across the dataset.

### 2. Volatility and range features

For each series:

- rolling realized volatility over `5`, `10`, `20`, and `60` days
- rolling average true range or normalized range
- rolling high-low range over `5` and `20` days

### 3. Trend and distance features

For each series:

- distance from `10`, `20`, `50`, and `200` day moving averages
- moving-average slope proxies
- rolling z-score over `20` and `60` days
- rolling drawdown from trailing `20`, `60`, and `252` day highs

### 4. Volume features

For ETFs where volume is reliable:

- volume z-score over `20` days
- relative volume vs trailing `20` day average
- price move times relative volume

Do not over-trust volume on day one of the project, but include it.

### 5. Risk-on / risk-off ratio features

These are likely to matter more than many raw series.

Start with:

- `QQQ / SPY`
- `IWM / SPY`
- `RSP / SPY`
- `XLY / XLP`
- `XLK / XLU`
- `XLF / XLU`
- `SMH / SPY`
- `HYG / IEF`
- `SPY / TLT`
- `GLD / UUP`

For each ratio, compute the same return, z-score, trend, and drawdown features as above.

### 6. Volatility regime features

From `^VIX`:

- level
- `1d`, `5d`, and `20d` changes
- z-score over `20` and `60` days
- distance from moving averages

Add explicit event flags derived from data known by the close of day `t`:

- `vix_up_10pct_flag`: `1` when `^VIX` close-to-close change is greater than or equal to `+10%`
- `vix_down_10pct_flag`: `1` when `^VIX` close-to-close change is less than or equal to `-10%`
- `vix_abs_10pct_flag`: `1` when absolute close-to-close VIX change is at least `10%`
- `vix_up_20pct_flag`: `1` when `^VIX` close-to-close change is greater than or equal to `+20%`
- `vix_down_20pct_flag`: `1` when `^VIX` close-to-close change is less than or equal to `-20%`
- `vix_abs_20pct_flag`: `1` when absolute close-to-close VIX change is at least `20%`

Optional later additions:

- `vix_up_5pct_flag`
- `vix_abs_5pct_flag`
- rolling count of VIX shock days over `5`, `10`, and `20` sessions
- interaction flags such as `vix_up_10pct_flag` plus `SPY` down day

These flags are useful because they mark unusual volatility-regime shifts that may deserve separate handling.

Use them as a tiered regime system:

- `10%` flags mark meaningful volatility shocks
- `20%` flags mark extreme stress or panic-like conditions

But use them carefully:

- include them as features,
- use them for sliced evaluation,
- and do not immediately overweight them in training.

The first goal is to let the model learn whether these event markers add signal.

Only after we have a stable baseline should we test whether event-day sample weighting improves results without hurting overall calibration.

### 7. Calendar features

Keep this small:

- day of week
- month
- month-end proximity
- quarter-end proximity

These are low priority, but cheap to include.

## Optional Phase 1.5 - SPY Options Features

Yes, SPY options can be useful.

But do **not** make the raw SPY options chain part of phase 1.

Recommendation:

- keep phase 1 as ETF and macro-proxy features,
- use `^VIX` as the first implied-volatility proxy,
- and only add SPY options data after the base pipeline is working.

Why I am drawing that line:

- SPY has many short-dated expirations, including daily expirations, which makes the raw chain high-dimensional and unstable for a first model.
- option chains shift constantly across strikes and maturities, so naive chain snapshots create ugly feature-engineering problems fast.
- open interest is useful for end-of-day research, but the full chain is easy to overfit.
- implied volatility and Greeks are more sensitive to snapshot timing and data quality than ETF OHLCV data.

If we add SPY options, add **aggregated** features first, not the full chain.

### Good first SPY options features

If a reliable end-of-day source is available, start with:

- total call volume
- total put volume
- put-call volume ratio
- total call open interest
- total put open interest
- put-call open interest ratio
- near-dated volume share, such as `0 to 7 DTE`
- short-dated versus medium-dated open interest, such as `0 to 7 DTE` versus `8 to 30 DTE`
- at-the-money implied volatility for a standard maturity bucket, if sourced cleanly
- simple term-structure slope, such as near-dated IV minus 1-month IV
- simple skew measure, such as put IV minus call IV at similar deltas, if sourced cleanly

### What not to do first

Do not start with:

- the full strike-by-strike chain as model input
- raw 0DTE flow features
- hand-built gamma-wall or max-pain features
- unvalidated Greeks from unstable free snapshots
- highly discretionary options-derived indicators

### Decision rule

Only add SPY options features when at least one of these is true:

- the base ETF-only model is working and we want an incremental feature block,
- we have a stable end-of-day options data source,
- and we can document exactly when the options snapshot becomes available for use.

If our only source is an inconsistent free chain scrape, skip options for now.

## What To Exclude From Phase 1

Do not include these yet:

- all S&P 500 constituent stocks
- fundamentals
- earnings events
- news or sentiment features
- options chain data
- macro releases with complex publication lags
- alternative data
- hand-picked indicator soup

Those are good ways to create complexity before we have proof that the pipeline works.

## Target Definition

### Primary target

Use future SPY adjusted-close return from `t` to `t+5`.

Recommended first label setup:

- `down`
- `flat`
- `up`

Use either:

- fixed thresholds such as `[-0.75%, +0.75%]`, or
- train-set quantile thresholds

I recommend starting with train-set quantiles so classes stay reasonably balanced.

### Secondary target

Use future SPY adjusted-close return from `t` to `t+1`.

This is mainly for benchmarking and sanity checks.

### Signal strength

Do not create a fake confidence score.

Instead define strength from the calibrated output distribution, for example:

- top-class probability,
- `P(up) - P(down)`,
- or `P(ret_5d > threshold)`.

## Sample Construction

Each training row is one trading day.

Each row should contain:

- all features computed using data available by the close of day `t`,
- the target for `t+1`,
- the target for `t+5`,
- and metadata such as the date and split assignment.

Drop rows where any required lookback window or forward target is not available.

## First Split Strategy

Use only time-aware splits.

Recommended first split:

- train: oldest `70%`
- validation: next `15%`
- test: most recent `15%`

After the first working result, move to walk-forward validation.

Do not randomly shuffle rows.

## Event-Day Evaluation Slice

Because high-volatility days are especially interesting, evaluate results both:

- on the full test set,
- on the `10%` VIX event subset such as `vix_abs_10pct_flag == 1`,
- and on the stricter `20%` VIX event subset such as `vix_abs_20pct_flag == 1`.

Report at least:

- class accuracy or directional hit rate,
- log loss,
- calibration quality,
- and average predicted signal strength on event days versus normal days.

This lets us learn whether the model is genuinely better during stress events, or whether it only looks better because those days are easier or more imbalanced.

## First Baselines To Compare Against

Before trusting a transformer, compare against:

- majority-class classifier
- logistic regression
- gradient boosting or XGBoost-style tree model if available

If the transformer cannot beat simple baselines on calibration and directional usefulness, do not trust it.

## Planned Transformer Architecture Notes

When we move from baselines to the sequence model, treat these as architectural requirements:

- use a compact time-series transformer, not a giant generic language-model style stack,
- include a latent state or regime token that can capture slower regime shifts alongside faster local sequence patterns,
- let that latent state feed the final prediction head so regime information can influence the output distribution,
- and make the final head output a probability distribution over the future return buckets rather than a single raw point prediction.

If we later add explicit regime labels, that latent state can also support an auxiliary regime objective, but the primary output should still be the future-move probability distribution.

## Acceptance Criteria For The Dataset

The dataset is good enough for the first model when:

- dates are aligned across all series,
- no feature uses future data,
- target creation is clearly documented,
- class counts are reasonable,
- missing-data handling is explicit,
- and a baseline model can train end-to-end on it.

## Planned Phase 2 Expansion

Only after phase 1 works, consider adding:

- constituent breadth features from S&P 500 members or a liquid proxy universe,
- advance/decline style measures,
- stronger cross-sectional internals,
- macro series with proper release lags,
- and more detailed regime labels.

Phase 2 should only happen after the first pipeline is reproducible and evaluated honestly.
