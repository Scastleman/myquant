from __future__ import annotations

import argparse
from datetime import date
import time

import pandas as pd
import yfinance as yf

from .download import RAW_PRICE_COLUMNS, _extract_single_ticker_frame
from .io import LARGE_UNIVERSE_MEMBERSHIP_PATH, LARGE_UNIVERSE_RAW_PRICES_PATH, write_parquet
from .large_universe_config import load_large_universe_config
from .universe import fetch_current_sp500_constituents


def _batched(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def resolve_large_universe_membership(
    *,
    wikipedia_url: str,
    stock_limit: int | None,
    context_tickers: tuple[str, ...],
) -> pd.DataFrame:
    constituents = fetch_current_sp500_constituents(wikipedia_url)
    if stock_limit is not None:
        constituents = constituents.head(stock_limit).copy()

    constituents["asset_type"] = "equity"
    context_frame = pd.DataFrame(
        {
            "symbol": list(context_tickers),
            "security": list(context_tickers),
            "gics_sector": ["context"] * len(context_tickers),
            "gics_sub_industry": ["context"] * len(context_tickers),
            "ticker": list(context_tickers),
            "asset_type": ["context"] * len(context_tickers),
        }
    )
    membership = pd.concat([constituents, context_frame], ignore_index=True)
    membership = membership.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return membership


def download_large_universe_price_history(
    *,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None,
    batch_size: int,
    max_retries: int,
    retry_sleep_seconds: float,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    batches = _batched(list(tickers), batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        print(f"Downloading batch {batch_index}/{len(batches)} with {len(batch)} tickers...", flush=True)
        downloaded = None
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                downloaded = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                if downloaded.empty:
                    raise RuntimeError("No data returned for batch.")
                break
            except Exception as exc:  # pragma: no cover - network reliability branch
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(retry_sleep_seconds * attempt)
        if downloaded is None or downloaded.empty:
            failed_tickers.extend(batch)
            if last_error is not None:
                print(f"  Batch failed after retries: {last_error}", flush=True)
            continue

        for ticker in batch:
            try:
                ticker_frame = _extract_single_ticker_frame(downloaded, ticker)
            except Exception:
                failed_tickers.append(ticker)
                continue
            if ticker_frame.empty or ticker_frame["adj_close"].isna().all():
                failed_tickers.append(ticker)
                continue
            frames.append(ticker_frame)

    if not frames:
        raise RuntimeError("No large-universe price data could be downloaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined = combined.loc[:, RAW_PRICE_COLUMNS]
    combined = combined.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"])
    return combined.reset_index(drop=True), sorted(set(failed_tickers))


def save_large_universe_raw_data(
    *,
    config_path: str = "configs/large_universe.toml",
    membership_output_path: str | None = None,
    prices_output_path: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_large_universe_config(config_path)
    membership = resolve_large_universe_membership(
        wikipedia_url=config.download.wikipedia_url,
        stock_limit=config.download.stock_limit,
        context_tickers=config.universe.context_tickers,
    )
    raw_prices, failed_tickers = download_large_universe_price_history(
        tickers=tuple(membership["ticker"].tolist()),
        start_date=config.download.start_date,
        end_date=end_date or config.download.end_date,
        batch_size=config.download.batch_size,
        max_retries=config.download.max_retries,
        retry_sleep_seconds=config.download.retry_sleep_seconds,
    )
    membership["has_price_data"] = membership["ticker"].isin(set(raw_prices["ticker"].unique()))
    membership["download_failed"] = membership["ticker"].isin(set(failed_tickers))

    write_parquet(membership, membership_output_path or LARGE_UNIVERSE_MEMBERSHIP_PATH, index=False)
    write_parquet(raw_prices, prices_output_path or LARGE_UNIVERSE_RAW_PRICES_PATH, index=False)
    return membership, raw_prices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a large daily current-500-plus-context universe.")
    parser.add_argument(
        "--config",
        default="configs/large_universe.toml",
        help="Path to the large-universe TOML config.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--membership-output",
        default=None,
        help="Optional parquet output path for membership metadata.",
    )
    parser.add_argument(
        "--prices-output",
        default=None,
        help="Optional parquet output path for raw prices.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    membership, raw_prices = save_large_universe_raw_data(
        config_path=args.config,
        membership_output_path=args.membership_output,
        prices_output_path=args.prices_output,
        end_date=args.end_date or date.today().isoformat(),
    )
    print(
        f"Saved membership for {len(membership):,} tickers and {len(raw_prices):,} raw rows "
        f"to {args.membership_output or LARGE_UNIVERSE_MEMBERSHIP_PATH} and "
        f"{args.prices_output or LARGE_UNIVERSE_RAW_PRICES_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
