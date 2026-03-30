from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import duckdb
import pandas as pd

from myquant.data.io import INTRADAY_BAR_STORE_ROOT, ensure_dir


REQUIRED_BAR_COLUMNS = ("timestamp", "ticker", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
TIMEFRAME_ALIASES = {
    "1m": "1min",
    "1min": "1min",
    "5m": "5min",
    "5min": "5min",
    "15m": "15min",
    "15min": "15min",
    "60m": "1h",
    "1h": "1h",
    "1d": "1d",
    "1day": "1d",
    "daily": "1d",
}


@dataclass(frozen=True)
class PartitionWriteResult:
    source: str
    timeframe: str
    session_date: str
    row_count: int
    ticker_count: int
    min_timestamp: str
    max_timestamp: str
    data_path: str
    metadata_path: str


def normalize_source_name(source: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", source.strip().lower())
    return slug.strip("_")


def normalize_timeframe_name(timeframe: str) -> str:
    normalized = timeframe.strip().lower().replace(" ", "")
    return TIMEFRAME_ALIASES.get(normalized, normalized)


def _coerce_utc_timestamp(series: pd.Series, input_timezone: str) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="raise")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            input_timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        timestamps = timestamps.dt.tz_convert(input_timezone)
    if timestamps.isna().any():
        raise ValueError("Timestamp localization introduced null values.")
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None)


def standardize_bar_frame(
    frame: pd.DataFrame,
    *,
    input_timezone: str,
    partition_timezone: str,
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required bar columns: {missing}")
    if frame.empty:
        raise ValueError("Bar frame is empty.")

    standardized = frame.copy()
    standardized["timestamp"] = _coerce_utc_timestamp(standardized["timestamp"], input_timezone=input_timezone)
    standardized["ticker"] = standardized["ticker"].astype(str).str.strip().str.upper()
    if (standardized["ticker"] == "").any():
        raise ValueError("Ticker symbols must be non-empty.")

    for column in PRICE_COLUMNS + ("volume",):
        standardized[column] = pd.to_numeric(standardized[column], errors="raise")

    session_timestamps = standardized["timestamp"].dt.tz_localize("UTC").dt.tz_convert(partition_timezone)
    standardized["session_date"] = session_timestamps.dt.strftime("%Y-%m-%d")
    standardized = standardized.sort_values(["timestamp", "ticker"]).drop_duplicates(
        subset=["timestamp", "ticker"],
        keep="last",
    )
    return standardized.reset_index(drop=True)


def build_bar_partition_path(
    *,
    root: str | Path,
    source: str,
    timeframe: str,
    session_date: str,
) -> Path:
    root_path = Path(root)
    trade_date = pd.Timestamp(session_date)
    return (
        root_path
        / f"source={normalize_source_name(source)}"
        / f"timeframe={normalize_timeframe_name(timeframe)}"
        / f"year={trade_date:%Y}"
        / f"month={trade_date:%m}"
        / f"date={trade_date:%Y-%m-%d}"
        / "bars.parquet"
    )


def _write_partition_file(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)


def _write_partition_metadata(
    *,
    frame: pd.DataFrame,
    output_path: Path,
    source: str,
    timeframe: str,
    session_date: str,
) -> PartitionWriteResult:
    metadata_path = output_path.with_name("metadata.json")
    result = PartitionWriteResult(
        source=normalize_source_name(source),
        timeframe=normalize_timeframe_name(timeframe),
        session_date=session_date,
        row_count=int(len(frame)),
        ticker_count=int(frame["ticker"].nunique()),
        min_timestamp=str(frame["timestamp"].min()),
        max_timestamp=str(frame["timestamp"].max()),
        data_path=str(output_path),
        metadata_path=str(metadata_path),
    )
    metadata_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def write_bar_batch(
    frame: pd.DataFrame,
    *,
    source: str,
    timeframe: str,
    root: str | Path = INTRADAY_BAR_STORE_ROOT,
    input_timezone: str = "America/New_York",
    partition_timezone: str = "America/New_York",
) -> list[PartitionWriteResult]:
    standardized = standardize_bar_frame(
        frame,
        input_timezone=input_timezone,
        partition_timezone=partition_timezone,
    )
    results: list[PartitionWriteResult] = []

    for session_date, partition_frame in standardized.groupby("session_date", sort=True):
        output_path = build_bar_partition_path(
            root=root,
            source=source,
            timeframe=timeframe,
            session_date=str(session_date),
        )
        write_frame = partition_frame.drop(columns=["session_date"]).reset_index(drop=True)
        _write_partition_file(write_frame, output_path)
        result = _write_partition_metadata(
            frame=write_frame,
            output_path=output_path,
            source=source,
            timeframe=timeframe,
            session_date=str(session_date),
        )
        results.append(result)

    return results


def _bar_store_glob(root: str | Path = INTRADAY_BAR_STORE_ROOT) -> str:
    return (Path(root) / "source=*" / "timeframe=*" / "year=*" / "month=*" / "date=*" / "bars.parquet").as_posix()


def list_bar_files(root: str | Path = INTRADAY_BAR_STORE_ROOT) -> list[Path]:
    return sorted(Path(root).rglob("bars.parquet"))


def register_bar_store_view(
    connection: duckdb.DuckDBPyConnection,
    *,
    root: str | Path = INTRADAY_BAR_STORE_ROOT,
    view_name: str = "intraday_bars",
) -> None:
    pattern = _bar_store_glob(root)
    escaped_pattern = pattern.replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"SELECT * FROM read_parquet('{escaped_pattern}', hive_partitioning=true, union_by_name=true)"
    )


def query_bar_store(
    *,
    root: str | Path = INTRADAY_BAR_STORE_ROOT,
    tickers: list[str] | None = None,
    timeframe: str | None = None,
    source: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    if not list_bar_files(root):
        return pd.DataFrame()

    owns_connection = connection is None
    con = connection or duckdb.connect()
    try:
        register_bar_store_view(con, root=root)
        clauses: list[str] = []
        params: list[object] = []

        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            clauses.append(f"ticker IN ({placeholders})")
            params.extend([ticker.upper() for ticker in tickers])
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(normalize_timeframe_name(timeframe))
        if source:
            clauses.append("source = ?")
            params.append(normalize_source_name(source))
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            clauses.append("timestamp < ?")
            params.append(pd.Timestamp(end))

        sql = "SELECT * FROM intraday_bars"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp, ticker"
        return con.execute(sql, params).df()
    finally:
        if owns_connection:
            con.close()


def summarize_bar_store(root: str | Path = INTRADAY_BAR_STORE_ROOT) -> pd.DataFrame:
    metadata_rows: list[dict] = []
    for metadata_path in sorted(Path(root).rglob("metadata.json")):
        metadata_rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    if not metadata_rows:
        return pd.DataFrame(
            columns=[
                "source",
                "timeframe",
                "session_date",
                "row_count",
                "ticker_count",
                "min_timestamp",
                "max_timestamp",
                "data_path",
                "metadata_path",
            ]
        )
    summary = pd.DataFrame.from_records(metadata_rows)
    return summary.sort_values(["source", "timeframe", "session_date"]).reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the intraday bar storage lake.")
    parser.add_argument(
        "--root",
        default=str(INTRADAY_BAR_STORE_ROOT),
        help="Intraday bar store root path.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of written partitions.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = summarize_bar_store(args.root)
    if summary.empty:
        print(f"No intraday bar partitions found under {args.root}")
        return

    print(
        f"Partitions={len(summary):,} | rows={int(summary['row_count'].sum()):,} | "
        f"sources={summary['source'].nunique()} | timeframes={summary['timeframe'].nunique()}",
    )
    print(summary.loc[:, ["source", "timeframe", "session_date", "row_count", "ticker_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
