from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from myquant.config import ProjectConfig, load_project_config
from myquant.features.market_features import (
    SplitFractions,
    assign_time_splits,
    compute_calendar_features,
    compute_drawdown_features,
    compute_ma_distance_features,
    compute_ratio_prices,
    compute_return_features,
    compute_volatility_features,
    compute_zscore_features,
)
from myquant.features.targets import (
    compute_future_return,
    fit_ternary_quantile_thresholds,
    label_ternary_returns,
)
from myquant.features.vix_events import add_vix_event_flags

from .dataset import (
    DRAWDOWN_WINDOWS,
    PRICE_WINDOWS,
    TREND_WINDOWS,
    VOL_WINDOWS,
    ZSCORE_WINDOWS,
    pivot_price_field,
)
from .io import PANEL_DATASET_PATH, RAW_PRICES_PATH, read_parquet, write_parquet


def _sanitize_ticker_name(ticker: str) -> str:
    return (
        ticker.replace("^", "caret_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _build_shared_feature_frame(raw_prices: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pivot_price_field(raw_prices, config.data.price_field)
    prices = prices.loc[:, list(config.universe.tickers)]
    prices = prices.dropna(how="any")

    ratio_prices = compute_ratio_prices(prices, config.universe.ratios)
    feature_frames = [
        compute_return_features(prices, PRICE_WINDOWS),
        compute_volatility_features(prices, VOL_WINDOWS),
        compute_ma_distance_features(prices, TREND_WINDOWS),
        compute_zscore_features(prices, ZSCORE_WINDOWS),
        compute_drawdown_features(prices, DRAWDOWN_WINDOWS),
        compute_return_features(ratio_prices, PRICE_WINDOWS),
        compute_zscore_features(ratio_prices, ZSCORE_WINDOWS),
        compute_drawdown_features(ratio_prices, DRAWDOWN_WINDOWS),
        compute_calendar_features(prices.index),
    ]

    vix_prices = prices.loc[:, ["^VIX"]].rename(columns={"^VIX": "adj_close"})
    vix_flags = add_vix_event_flags(
        vix_prices,
        close_column="adj_close",
        thresholds=config.events.vix_thresholds,
    ).drop(columns=["adj_close"])

    shared_frame = pd.concat([*feature_frames, vix_flags], axis=1).dropna()
    shared_frame["split"] = assign_time_splits(shared_frame.index, fractions=SplitFractions())
    return prices, shared_frame


def build_panel_dataset(raw_prices: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    prices, shared_frame = _build_shared_feature_frame(raw_prices, config)

    panel_frames: list[pd.DataFrame] = []
    target_tickers = tuple(config.panel_training.target_tickers)
    primary_horizon = config.targets.primary_horizon_days
    benchmark_horizon = config.targets.benchmark_horizon_days

    for ticker in target_tickers:
        primary_return = compute_future_return(prices[ticker], primary_horizon).reindex(shared_frame.index)
        benchmark_return = compute_future_return(prices[ticker], benchmark_horizon).reindex(shared_frame.index)
        lower_threshold, upper_threshold = fit_ternary_quantile_thresholds(
            primary_return.loc[shared_frame["split"] == "train"],
            quantiles=config.targets.train_quantiles,
        )

        ticker_frame = shared_frame.copy()
        ticker_frame["target_ticker"] = ticker
        ticker_frame[f"target_ret_{primary_horizon}d"] = primary_return
        ticker_frame[f"target_ret_{benchmark_horizon}d"] = benchmark_return
        ticker_frame[f"target_label_{primary_horizon}d"] = label_ternary_returns(
            primary_return,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        ticker_frame["target_lower_threshold"] = lower_threshold
        ticker_frame["target_upper_threshold"] = upper_threshold
        panel_frames.append(ticker_frame.dropna())

    panel = pd.concat(panel_frames, axis=0)
    panel["is_focus_ticker"] = (
        panel["target_ticker"] == config.panel_training.evaluation_focus_ticker
    ).astype(np.int8)
    for ticker in target_tickers:
        panel[f"target_ticker_{_sanitize_ticker_name(ticker)}"] = (
            panel["target_ticker"] == ticker
        ).astype(np.int8)

    panel = panel.reset_index(names="date")
    panel = panel.sort_values(["target_ticker", "date"]).reset_index(drop=True)
    return panel


def save_panel_dataset(
    config: ProjectConfig,
    raw_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    raw_prices = read_parquet(raw_path or RAW_PRICES_PATH)
    dataset = build_panel_dataset(raw_prices=raw_prices, config=config)
    write_parquet(dataset, output_path or PANEL_DATASET_PATH, index=False)
    return dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the grouped panel dataset for multi-target training.")
    parser.add_argument(
        "--config",
        default="configs/project.toml",
        help="Path to the project TOML config.",
    )
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Optional raw parquet path. Defaults to data/raw/prices.parquet.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional processed parquet output path.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_project_config(args.config)
    dataset = save_panel_dataset(
        config=config,
        raw_path=args.raw_path,
        output_path=args.output,
    )
    print(
        f"Saved {len(dataset):,} panel rows and {len(dataset.columns):,} columns "
        f"to {args.output or PANEL_DATASET_PATH}",
    )


if __name__ == "__main__":
    main()
