from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from myquant.config.settings import RatioSpec


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if not np.isclose(total, 1.0):
            raise ValueError("Split fractions must sum to 1.0")


def assign_time_splits(
    index: pd.Index,
    fractions: SplitFractions = SplitFractions(),
) -> pd.Series:
    """Assign train/validation/test splits in chronological order."""
    n_rows = len(index)
    if n_rows < 3:
        raise ValueError("Need at least three rows to assign time splits")

    train_end = max(1, int(np.floor(n_rows * fractions.train)))
    validation_end = max(train_end + 1, int(np.floor(n_rows * (fractions.train + fractions.validation))))
    validation_end = min(validation_end, n_rows - 1)

    split = pd.Series(index=index, dtype="object")
    split.iloc[:train_end] = "train"
    split.iloc[train_end:validation_end] = "validation"
    split.iloc[validation_end:] = "test"
    return split


def compute_ratio_prices(
    prices: pd.DataFrame,
    ratio_specs: tuple[RatioSpec, ...],
) -> pd.DataFrame:
    """Compute named ratio series from a wide price table."""
    ratio_data: dict[str, pd.Series] = {}
    for spec in ratio_specs:
        ratio_data[spec.name] = prices[spec.numerator].div(prices[spec.denominator])
    return pd.DataFrame(ratio_data, index=prices.index)


def _feature_columns(frame: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    return frame.rename(columns=lambda column: f"{column}__{feature_name}")


def compute_return_features(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for window in windows:
        returns = prices.pct_change(window)
        frames.append(_feature_columns(returns, f"ret_{window}d"))
    return pd.concat(frames, axis=1)


def compute_volatility_features(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    daily_returns = prices.pct_change()
    frames = []
    for window in windows:
        rolling_vol = daily_returns.rolling(window).std()
        frames.append(_feature_columns(rolling_vol, f"vol_{window}d"))
    return pd.concat(frames, axis=1)


def compute_ma_distance_features(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for window in windows:
        ma = prices.rolling(window).mean()
        distance = prices.div(ma).sub(1.0)
        frames.append(_feature_columns(distance, f"dist_ma_{window}d"))
    return pd.concat(frames, axis=1)


def compute_zscore_features(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for window in windows:
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        zscore = prices.sub(rolling_mean).div(rolling_std.replace(0.0, np.nan))
        frames.append(_feature_columns(zscore, f"zscore_{window}d"))
    return pd.concat(frames, axis=1)


def compute_drawdown_features(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for window in windows:
        rolling_max = prices.rolling(window).max()
        drawdown = prices.div(rolling_max).sub(1.0)
        frames.append(_feature_columns(drawdown, f"drawdown_{window}d"))
    return pd.concat(frames, axis=1)


def compute_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a small set of low-cost calendar features."""
    frame = pd.DataFrame(index=index)
    frame["calendar__day_of_week"] = index.dayofweek
    frame["calendar__month"] = index.month
    month_end_distance = (index.to_period("M").to_timestamp("M") - index).days
    quarter_end_distance = (index.to_period("Q").to_timestamp("Q") - index).days
    frame["calendar__days_to_month_end"] = month_end_distance.astype("int16")
    frame["calendar__days_to_quarter_end"] = quarter_end_distance.astype("int16")
    return frame
