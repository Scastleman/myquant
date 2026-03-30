# Intraday Storage

This project now uses a simple but scalable storage design for high-frequency market data:

- raw bars live in a partitioned Parquet lake,
- DuckDB is the query layer,
- and partitions are written by `source x timeframe x session_date`.

## Why This Shape

We are explicitly **not** partitioning by ticker.

That sounds tempting at first, but it explodes file counts once we move to:

- many assets,
- multiple timeframes,
- and multi-year history.

The storage unit we want is:

- one daily partition for one source and one timeframe,
- containing all tickers for that session.

That keeps overwrite behavior simple, avoids millions of tiny files, and still lets DuckDB prune partitions well.

## Layout

```text
data/lake/bars/
  source=polygon/
    timeframe=1min/
      year=2026/
        month=03/
          date=2026-03-27/
            bars.parquet
            metadata.json
```

## Raw Schema

Each `bars.parquet` file stores rows in long form with at least:

- `timestamp`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `volume`

Timestamps are normalized to naive UTC inside the file. The partition `date=...` is derived in the configured market/session timezone, which defaults to `America/New_York`.

## Metadata Sidecars

Each partition also gets a `metadata.json` file with:

- source
- timeframe
- session date
- row count
- ticker count
- min/max timestamps

That lets us inspect the lake without scanning all parquet files.

## Query Layer

DuckDB reads the partitioned parquet tree directly with hive partition discovery. That means `source`, `timeframe`, `year`, `month`, and `date` are available for filtering without duplicating them into every row.

## What Comes Next

This solves the storage shape, not the provider problem.

The next step is to build an intraday provider adapter that writes into this lake format. Once we do that, the research pipeline can:

1. read raw intraday bars from DuckDB,
2. resample and compute features,
3. build multi-timeframe windows,
4. and train on a much larger sequence budget.
