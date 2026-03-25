from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import yfinance as yf

from myquant.config import load_project_config

from .io import RAW_PRICES_PATH, write_parquet


RAW_PRICE_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize a ticker for filenames and column prefixes when needed."""
    return ticker.replace("^", "").replace("=", "_").replace("-", "_")


def _rename_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" not in renamed.columns:
        renamed["adj_close"] = renamed["close"]
    return renamed


def _extract_single_ticker_frame(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(downloaded.columns, pd.MultiIndex):
        ticker_frame = downloaded[ticker].copy()
    else:
        ticker_frame = downloaded.copy()

    ticker_frame = _rename_price_columns(ticker_frame)
    ticker_frame = ticker_frame.reset_index(names="date")
    ticker_frame["ticker"] = ticker
    return ticker_frame.loc[:, RAW_PRICE_COLUMNS]


def download_price_history(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download the configured daily OHLCV history into one long-form frame."""
    downloaded = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if downloaded.empty:
        raise RuntimeError("No data was returned by yfinance.")

    frames = [_extract_single_ticker_frame(downloaded, ticker) for ticker in tickers]
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined = combined.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"])
    return combined.reset_index(drop=True)


def save_raw_prices(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Download and persist the raw daily price history."""
    raw_prices = download_price_history(tickers=tickers, start_date=start_date, end_date=end_date)
    write_parquet(raw_prices, output_path or RAW_PRICES_PATH, index=False)
    return raw_prices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download raw daily price history for myquant.")
    parser.add_argument(
        "--config",
        default="configs/project.toml",
        help="Path to the project TOML config.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional parquet output path. Defaults to data/raw/prices.parquet.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_project_config(args.config)
    raw_prices = save_raw_prices(
        tickers=config.universe.tickers,
        start_date=config.project.data_start_date,
        end_date=args.end_date or date.today().isoformat(),
        output_path=args.output,
    )
    print(
        f"Saved {len(raw_prices):,} raw rows for {len(config.universe.tickers)} tickers "
        f"to {args.output or RAW_PRICES_PATH}",
    )


if __name__ == "__main__":
    main()
