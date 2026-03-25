# Pre-Build Checklist

This is the minimum setup to finish before writing the main pipeline code.

## Must Do First

### 1. Lock the Python version

Do not build the project on Python `3.14` unless package compatibility is proven.

Recommended:

- Python `3.12` as the default choice
- Python `3.13` if the libraries we need are confirmed stable

Reason:

- quant, data, and deep-learning stacks often lag the newest Python release
- losing time to package compatibility is not part of the learning goal

### 2. Define the first historical window

Choose a default raw-data start date before coding.

Recommended:

- start around `2006-01-01`

Reason:

- this gives us multiple major regimes
- it covers 2008, 2020, 2022, and calmer periods
- it is long enough for rolling features like `252d`

### 3. Freeze the first target spec

Before coding, write the exact label rule we will use.

Recommended first target:

- primary target: SPY adjusted-close return from `t` to `t+5`
- label shape: `down / flat / up`
- thresholds: train-set quantiles
- secondary benchmark: SPY direction from `t` to `t+1`

### 4. Freeze the first evaluation metrics

Do not wait until after training to decide what counts as success.

Recommended first metrics:

- log loss
- balanced accuracy
- Brier score
- calibration curve / reliability summary
- directional hit rate for `up` vs `down`
- event-day metrics on VIX `10%` and `20%` subsets

### 5. Pick the first data access library

Recommended:

- use `yfinance` for the first pass

Reason:

- fast to start
- enough for an exploratory project
- easy to swap later if needed

## Strongly Recommended Before Coding

### 6. Decide the repo structure

Recommended top-level layout:

- `src/`
- `data/`
- `notebooks/`
- `configs/`
- `artifacts/`
- `tests/`

### 7. Decide the first file formats

Recommended:

- raw downloads: Parquet
- engineered dataset: Parquet
- run metadata and metrics: JSON or CSV

Do not start with Postgres.

### 8. Decide the normalization rule

Recommended:

- fit normalization only on the training split
- apply the fitted transform to validation and test
- persist normalization metadata with each run

### 9. Decide the split dates up front

Recommended first pass:

- train: oldest `70%`
- validation: next `15%`
- test: most recent `15%`

Then move to walk-forward evaluation after the first complete pipeline.

### 10. Decide the baseline models now

Recommended:

- majority-class classifier
- logistic regression
- gradient-boosted tree model

If a transformer cannot beat these honestly, do not trust it.

## Nice To Have

### 11. Start an experiment log

Keep a lightweight experiment journal with:

- run id
- feature set version
- target version
- split version
- model config
- headline metrics
- short notes on what changed

### 12. Make the first acceptance rule explicit

Recommended phase-1 success bar:

- dataset builds reproducibly
- baseline model trains end-to-end
- probabilities are at least somewhat calibrated
- test performance is better than naive baseline on at least one honest metric
- event-day slice is reported separately

## What Is Already Ready

We already have:

- the project behavior contract in `BASELINE_PROMPT.md`
- the first dataset definition in `FIRST_DATA_SPEC.md`

## Opinionated Summary

Before building, the only things I consider truly essential are:

- move off Python `3.14`
- freeze the first target and metrics
- choose the date range
- choose the initial storage and repo layout

After that, stop planning and start coding.
