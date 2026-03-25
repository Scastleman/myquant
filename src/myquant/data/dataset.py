from __future__ import annotations

import argparse

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

from .io import PROCESSED_DATASET_PATH, RAW_PRICES_PATH, read_parquet, write_parquet


PRICE_WINDOWS = (1, 3, 5, 10, 20, 60, 120, 252)
VOL_WINDOWS = (5, 10, 20, 60)
TREND_WINDOWS = (10, 20, 50, 200)
ZSCORE_WINDOWS = (20, 60)
DRAWDOWN_WINDOWS = (20, 60, 252)


def pivot_price_field(raw_prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot the long raw table into a wide date-by-ticker matrix for one field."""
    wide = raw_prices.pivot(index="date", columns="ticker", values=field)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def build_phase1_dataset(raw_prices: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Build the first processed dataset from the long raw price table."""
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

    primary_return = compute_future_return(prices["SPY"], config.targets.primary_horizon_days)
    benchmark_return = compute_future_return(prices["SPY"], config.targets.benchmark_horizon_days)

    dataset = pd.concat(
        [
            *feature_frames,
            vix_flags,
            pd.DataFrame(
                {
                    f"target_ret_{config.targets.primary_horizon_days}d": primary_return,
                    f"target_ret_{config.targets.benchmark_horizon_days}d": benchmark_return,
                },
                index=prices.index,
            ),
        ],
        axis=1,
    ).dropna()

    split = assign_time_splits(dataset.index, fractions=SplitFractions())
    lower, upper = fit_ternary_quantile_thresholds(
        dataset.loc[split == "train", f"target_ret_{config.targets.primary_horizon_days}d"],
        quantiles=config.targets.train_quantiles,
    )
    dataset["split"] = split
    dataset[f"target_label_{config.targets.primary_horizon_days}d"] = label_ternary_returns(
        dataset[f"target_ret_{config.targets.primary_horizon_days}d"],
        lower_threshold=lower,
        upper_threshold=upper,
    )
    dataset = dataset.reset_index(names="date")
    return dataset, (lower, upper)


def save_phase1_dataset(
    config: ProjectConfig,
    raw_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    raw_prices = read_parquet(raw_path or RAW_PRICES_PATH)
    dataset, thresholds = build_phase1_dataset(raw_prices=raw_prices, config=config)
    dataset["target_lower_threshold"] = thresholds[0]
    dataset["target_upper_threshold"] = thresholds[1]
    write_parquet(dataset, output_path or PROCESSED_DATASET_PATH, index=False)
    return dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the phase-1 processed dataset.")
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
    dataset = save_phase1_dataset(
        config=config,
        raw_path=args.raw_path,
        output_path=args.output,
    )
    print(
        f"Saved {len(dataset):,} processed rows and {len(dataset.columns):,} columns "
        f"to {args.output or PROCESSED_DATASET_PATH}",
    )


if __name__ == "__main__":
    main()
