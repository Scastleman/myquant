from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from myquant.config.settings import RatioSpec
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
from myquant.features.targets import compute_future_return, fit_ternary_quantile_thresholds, label_ternary_returns
from myquant.features.vix_events import add_vix_event_flags

from .dataset import DRAWDOWN_WINDOWS, TREND_WINDOWS, VOL_WINDOWS, ZSCORE_WINDOWS, pivot_price_field
from .io import (
    LARGE_UNIVERSE_MEMBERSHIP_PATH,
    LARGE_UNIVERSE_PANEL_DATASET_PATH,
    LARGE_UNIVERSE_RAW_PRICES_PATH,
    SPY_BREADTH_DATASET_PATH,
    read_parquet,
    write_parquet,
)
from .large_universe_config import LargeUniverseConfig, load_large_universe_config
from .universe import RATIO_SPECS


STOCK_RETURN_WINDOWS = (1, 5, 20, 60, 120, 252)
STOCK_VOL_WINDOWS = (20, 60)
STOCK_MA_WINDOWS = (20, 50, 200)
STOCK_ZSCORE_WINDOWS = (20, 60)
STOCK_DRAWDOWN_WINDOWS = (60, 252)
RELATIVE_RETURN_WINDOWS = (1, 5, 20, 60)
RELATIVE_MA_WINDOWS = (50, 200)
RELATIVE_ZSCORE_WINDOWS = (20, 60)
RELATIVE_DRAWDOWN_WINDOWS = (60, 252)
BREADTH_RETURN_WINDOWS = (1, 5, 20)
BREADTH_MA_WINDOWS = (50, 200)


def _stack_feature_frame(frame: pd.DataFrame, suffix: str) -> pd.Series:
    stacked = frame.stack()
    stacked.index.names = ["date", "target_ticker"]
    return stacked.rename(suffix)


def _slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("&", "and")
    )


def _build_context_features(
    context_prices: pd.DataFrame,
    *,
    vix_thresholds: tuple[float, ...],
) -> pd.DataFrame:
    available_tickers = set(context_prices.columns)
    ratio_specs = tuple(
        RatioSpec(*spec)
        for spec in RATIO_SPECS
        if spec[1] in available_tickers and spec[2] in available_tickers
    )
    ratio_prices = compute_ratio_prices(context_prices, ratio_specs)
    feature_frames = [
        compute_return_features(context_prices, STOCK_RETURN_WINDOWS),
        compute_volatility_features(context_prices, VOL_WINDOWS),
        compute_ma_distance_features(context_prices, TREND_WINDOWS),
        compute_zscore_features(context_prices, ZSCORE_WINDOWS),
        compute_drawdown_features(context_prices, DRAWDOWN_WINDOWS),
        compute_return_features(ratio_prices, STOCK_RETURN_WINDOWS),
        compute_zscore_features(ratio_prices, ZSCORE_WINDOWS),
        compute_drawdown_features(ratio_prices, DRAWDOWN_WINDOWS),
        compute_calendar_features(context_prices.index),
    ]

    vix_prices = context_prices.loc[:, ["^VIX"]].rename(columns={"^VIX": "adj_close"})
    vix_flags = add_vix_event_flags(
        vix_prices,
        close_column="adj_close",
        thresholds=vix_thresholds,
    ).drop(columns=["adj_close"])
    feature_frames.append(vix_flags)
    return pd.concat(feature_frames, axis=1)


def _build_breadth_features(stock_prices: pd.DataFrame, spy_prices: pd.Series) -> pd.DataFrame:
    breadth = pd.DataFrame(index=stock_prices.index)
    breadth["breadth__stock_count"] = stock_prices.notna().sum(axis=1).astype("int16")

    stock_return_1d = stock_prices.pct_change()
    spy_return_1d = spy_prices.pct_change()
    positive_mask = stock_return_1d.gt(0.0)
    negative_mask = stock_return_1d.lt(0.0)
    breadth["breadth__pct_positive_ret_1d"] = positive_mask.where(stock_return_1d.notna()).mean(axis=1)
    breadth["breadth__pct_negative_ret_1d"] = negative_mask.where(stock_return_1d.notna()).mean(axis=1)
    breadth["breadth__adv_minus_dec_1d"] = (
        breadth["breadth__pct_positive_ret_1d"] - breadth["breadth__pct_negative_ret_1d"]
    )

    for window in BREADTH_RETURN_WINDOWS:
        stock_ret = stock_prices.pct_change(window)
        spy_ret = spy_prices.pct_change(window)
        excess_ret = stock_ret.sub(spy_ret, axis=0)
        breadth[f"breadth__median_excess_ret_{window}d"] = excess_ret.median(axis=1, skipna=True)
        breadth[f"breadth__pct_outperform_spy_{window}d"] = excess_ret.gt(0.0).where(excess_ret.notna()).mean(axis=1)
        breadth[f"breadth__dispersion_ret_{window}d"] = stock_ret.std(axis=1, skipna=True)

    for window in BREADTH_MA_WINDOWS:
        ma = stock_prices.rolling(window).mean()
        above = stock_prices.gt(ma).where(stock_prices.notna() & ma.notna())
        breadth[f"breadth__pct_above_ma_{window}d"] = above.mean(axis=1)

    return breadth


def _build_stock_feature_panel(
    stock_prices: pd.DataFrame,
    spy_prices: pd.Series,
) -> pd.DataFrame:
    feature_series: list[pd.Series] = []

    stock_returns = {window: stock_prices.pct_change(window) for window in STOCK_RETURN_WINDOWS}
    stock_vols = {
        window: stock_prices.pct_change().rolling(window).std()
        for window in STOCK_VOL_WINDOWS
    }
    stock_ma_dist = {
        window: stock_prices.div(stock_prices.rolling(window).mean()).sub(1.0)
        for window in STOCK_MA_WINDOWS
    }
    stock_zscores = {
        window: stock_prices.sub(stock_prices.rolling(window).mean()).div(
            stock_prices.rolling(window).std().replace(0.0, np.nan)
        )
        for window in STOCK_ZSCORE_WINDOWS
    }
    stock_drawdowns = {
        window: stock_prices.div(stock_prices.rolling(window).max()).sub(1.0)
        for window in STOCK_DRAWDOWN_WINDOWS
    }

    spy_returns = {window: spy_prices.pct_change(window) for window in RELATIVE_RETURN_WINDOWS}
    relative_prices = stock_prices.div(spy_prices, axis=0)
    log_relative_prices = np.log(relative_prices.replace(0.0, np.nan))
    relative_returns = {
        window: stock_prices.pct_change(window).sub(spy_returns[window], axis=0)
        for window in RELATIVE_RETURN_WINDOWS
    }
    relative_ma_dist = {
        window: relative_prices.div(relative_prices.rolling(window).mean()).sub(1.0)
        for window in RELATIVE_MA_WINDOWS
    }
    relative_zscores = {
        window: relative_prices.sub(relative_prices.rolling(window).mean()).div(
            relative_prices.rolling(window).std().replace(0.0, np.nan)
        )
        for window in RELATIVE_ZSCORE_WINDOWS
    }
    relative_drawdowns = {
        window: relative_prices.div(relative_prices.rolling(window).max()).sub(1.0)
        for window in RELATIVE_DRAWDOWN_WINDOWS
    }

    for window, frame in stock_returns.items():
        feature_series.append(_stack_feature_frame(frame, f"stock__ret_{window}d"))
    for window, frame in stock_vols.items():
        feature_series.append(_stack_feature_frame(frame, f"stock__vol_{window}d"))
    for window, frame in stock_ma_dist.items():
        feature_series.append(_stack_feature_frame(frame, f"stock__dist_ma_{window}d"))
    for window, frame in stock_zscores.items():
        feature_series.append(_stack_feature_frame(frame, f"stock__zscore_{window}d"))
    for window, frame in stock_drawdowns.items():
        feature_series.append(_stack_feature_frame(frame, f"stock__drawdown_{window}d"))

    for window, frame in relative_returns.items():
        feature_series.append(_stack_feature_frame(frame, f"relative_spy__ret_diff_{window}d"))
    feature_series.append(_stack_feature_frame(log_relative_prices, "relative_spy__log_price_ratio"))
    for window, frame in relative_ma_dist.items():
        feature_series.append(_stack_feature_frame(frame, f"relative_spy__dist_ma_{window}d"))
    for window, frame in relative_zscores.items():
        feature_series.append(_stack_feature_frame(frame, f"relative_spy__zscore_{window}d"))
    for window, frame in relative_drawdowns.items():
        feature_series.append(_stack_feature_frame(frame, f"relative_spy__drawdown_{window}d"))

    panel = pd.concat(feature_series, axis=1).reset_index()
    return panel


def _build_sector_dummies(membership: pd.DataFrame) -> pd.DataFrame:
    stocks = membership.loc[membership["asset_type"] == "equity", ["ticker", "gics_sector"]].copy()
    dummies = pd.get_dummies(stocks["gics_sector"].map(_slugify), prefix="sector", dtype="int8")
    return pd.concat([stocks.rename(columns={"ticker": "target_ticker"}), dummies], axis=1)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "split", "target_ticker"}
    excluded_prefixes = ("target_",)

    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(column)
    return feature_columns


def build_large_universe_datasets(
    raw_prices: pd.DataFrame,
    membership: pd.DataFrame,
    config: LargeUniverseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = membership.copy()
    context_tickers = tuple(config.universe.context_tickers)
    available_tickers = set(raw_prices["ticker"].unique())
    missing_context = sorted(set(context_tickers) - available_tickers)
    if missing_context:
        raise ValueError(f"Raw prices are missing required context tickers: {missing_context}")

    context_prices = pivot_price_field(
        raw_prices.loc[raw_prices["ticker"].isin(context_tickers)].copy(),
        "adj_close",
    )
    context_prices = context_prices.loc[:, list(context_tickers)].dropna(how="any")

    stock_membership = membership.loc[
        (membership["asset_type"] == "equity") & (membership["has_price_data"]),
        :,
    ].copy()
    stock_prices = pivot_price_field(
        raw_prices.loc[raw_prices["ticker"].isin(stock_membership["ticker"])].copy(),
        "adj_close",
    ).reindex(context_prices.index)

    history_counts = stock_prices.notna().sum(axis=0)
    eligible_stocks = history_counts.loc[history_counts >= config.dataset.min_history_days].index.tolist()
    stock_prices = stock_prices.loc[:, eligible_stocks]
    stock_membership = stock_membership.loc[stock_membership["ticker"].isin(eligible_stocks)].copy()

    shared_context = _build_context_features(
        context_prices,
        vix_thresholds=config.dataset.vix_thresholds,
    )
    breadth_features = _build_breadth_features(stock_prices, context_prices["SPY"])
    shared_features = pd.concat([shared_context, breadth_features], axis=1)
    shared_features = shared_features.loc[
        shared_features["breadth__stock_count"] >= config.dataset.min_breadth_stock_count
    ].dropna().copy()
    shared_features["split"] = assign_time_splits(shared_features.index, fractions=SplitFractions())

    spy_prices = context_prices["SPY"]
    panel = _build_stock_feature_panel(stock_prices, spy_prices)
    panel = panel.merge(shared_features.reset_index(names="date"), on="date", how="inner")

    stock_future_ret_5d = compute_future_return(stock_prices, config.dataset.primary_horizon_days)
    stock_future_ret_1d = compute_future_return(stock_prices, config.dataset.benchmark_horizon_days)
    spy_future_ret_5d = compute_future_return(spy_prices, config.dataset.primary_horizon_days)
    spy_future_ret_1d = compute_future_return(spy_prices, config.dataset.benchmark_horizon_days)

    target_panel = pd.concat(
        [
            _stack_feature_frame(
                stock_future_ret_5d,
                f"target_ret_{config.dataset.primary_horizon_days}d",
            ),
            _stack_feature_frame(
                stock_future_ret_1d,
                f"target_ret_{config.dataset.benchmark_horizon_days}d",
            ),
            _stack_feature_frame(
                stock_future_ret_5d.sub(spy_future_ret_5d, axis=0),
                f"target_excess_ret_{config.dataset.primary_horizon_days}d",
            ),
            _stack_feature_frame(
                stock_future_ret_1d.sub(spy_future_ret_1d, axis=0),
                f"target_excess_ret_{config.dataset.benchmark_horizon_days}d",
            ),
        ],
        axis=1,
    ).reset_index()
    panel = panel.merge(target_panel, on=["date", "target_ticker"], how="left")

    sector_dummies = _build_sector_dummies(stock_membership)
    panel = panel.merge(sector_dummies, on="target_ticker", how="left")

    feature_columns = _feature_columns(panel)
    required = [
        *feature_columns,
        f"target_ret_{config.dataset.primary_horizon_days}d",
        f"target_ret_{config.dataset.benchmark_horizon_days}d",
        f"target_excess_ret_{config.dataset.primary_horizon_days}d",
        f"target_excess_ret_{config.dataset.benchmark_horizon_days}d",
    ]
    panel = panel.dropna(subset=required).reset_index(drop=True)

    lower, upper = fit_ternary_quantile_thresholds(
        panel.loc[
            panel["split"] == "train",
            f"target_ret_{config.dataset.primary_horizon_days}d",
        ],
        quantiles=config.dataset.train_quantiles,
    )
    excess_lower, excess_upper = fit_ternary_quantile_thresholds(
        panel.loc[
            panel["split"] == "train",
            f"target_excess_ret_{config.dataset.primary_horizon_days}d",
        ],
        quantiles=config.dataset.train_quantiles,
    )
    panel[f"target_label_{config.dataset.primary_horizon_days}d"] = label_ternary_returns(
        panel[f"target_ret_{config.dataset.primary_horizon_days}d"],
        lower_threshold=lower,
        upper_threshold=upper,
    )
    panel[f"target_excess_label_{config.dataset.primary_horizon_days}d"] = label_ternary_returns(
        panel[f"target_excess_ret_{config.dataset.primary_horizon_days}d"],
        lower_threshold=excess_lower,
        upper_threshold=excess_upper,
    )
    panel["target_lower_threshold"] = lower
    panel["target_upper_threshold"] = upper
    panel["target_excess_lower_threshold"] = excess_lower
    panel["target_excess_upper_threshold"] = excess_upper
    panel["deoverlap_group_5d"] = panel.groupby("target_ticker").cumcount().mod(config.dataset.primary_horizon_days)
    panel = panel.sort_values(["target_ticker", "date"]).reset_index(drop=True)

    spy_dataset = pd.concat(
        [
            shared_features,
            pd.DataFrame(
                {
                    f"target_ret_{config.dataset.primary_horizon_days}d": spy_future_ret_5d.reindex(shared_features.index),
                    f"target_ret_{config.dataset.benchmark_horizon_days}d": spy_future_ret_1d.reindex(shared_features.index),
                },
                index=shared_features.index,
            ),
        ],
        axis=1,
    ).dropna()
    spy_lower, spy_upper = fit_ternary_quantile_thresholds(
        spy_dataset.loc[
            spy_dataset["split"] == "train",
            f"target_ret_{config.dataset.primary_horizon_days}d",
        ],
        quantiles=config.dataset.train_quantiles,
    )
    spy_dataset[f"target_label_{config.dataset.primary_horizon_days}d"] = label_ternary_returns(
        spy_dataset[f"target_ret_{config.dataset.primary_horizon_days}d"],
        lower_threshold=spy_lower,
        upper_threshold=spy_upper,
    )
    spy_dataset["target_lower_threshold"] = spy_lower
    spy_dataset["target_upper_threshold"] = spy_upper
    spy_dataset["deoverlap_group_5d"] = np.arange(len(spy_dataset)) % config.dataset.primary_horizon_days
    spy_dataset = spy_dataset.reset_index(names="date")

    return panel, spy_dataset


def save_large_universe_datasets(
    *,
    config_path: str = "configs/large_universe.toml",
    raw_prices_path: str | None = None,
    membership_path: str | None = None,
    panel_output_path: str | None = None,
    spy_output_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_large_universe_config(config_path)
    raw_prices = read_parquet(raw_prices_path or LARGE_UNIVERSE_RAW_PRICES_PATH)
    membership = read_parquet(membership_path or LARGE_UNIVERSE_MEMBERSHIP_PATH)
    panel, spy_dataset = build_large_universe_datasets(raw_prices, membership, config)
    write_parquet(panel, panel_output_path or LARGE_UNIVERSE_PANEL_DATASET_PATH, index=False)
    write_parquet(spy_dataset, spy_output_path or SPY_BREADTH_DATASET_PATH, index=False)
    return panel, spy_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the large-universe daily stock panel and SPY breadth dataset.")
    parser.add_argument(
        "--config",
        default="configs/large_universe.toml",
        help="Path to the large-universe TOML config.",
    )
    parser.add_argument(
        "--raw-prices-path",
        default=None,
        help="Optional raw large-universe prices parquet path.",
    )
    parser.add_argument(
        "--membership-path",
        default=None,
        help="Optional membership parquet path.",
    )
    parser.add_argument(
        "--panel-output",
        default=None,
        help="Optional output path for the stock panel dataset.",
    )
    parser.add_argument(
        "--spy-output",
        default=None,
        help="Optional output path for the SPY breadth dataset.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    panel, spy_dataset = save_large_universe_datasets(
        config_path=args.config,
        raw_prices_path=args.raw_prices_path,
        membership_path=args.membership_path,
        panel_output_path=args.panel_output,
        spy_output_path=args.spy_output,
    )
    print(
        f"Saved {len(panel):,} stock-panel rows and {len(spy_dataset):,} SPY breadth rows "
        f"to {args.panel_output or LARGE_UNIVERSE_PANEL_DATASET_PATH} and "
        f"{args.spy_output or SPY_BREADTH_DATASET_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
