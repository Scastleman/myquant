from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def compute_future_return(prices: pd.Series, horizon: int) -> pd.Series:
    """Compute the forward percentage return from t to t + horizon."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return prices.shift(-horizon).div(prices).sub(1.0)


def fit_ternary_quantile_thresholds(
    future_returns: pd.Series,
    quantiles: Iterable[float] = (1.0 / 3.0, 2.0 / 3.0),
) -> tuple[float, float]:
    """Fit lower and upper quantile thresholds on the provided sample."""
    lower_q, upper_q = tuple(quantiles)
    if not 0.0 < lower_q < upper_q < 1.0:
        raise ValueError("quantiles must satisfy 0.0 < lower < upper < 1.0")

    clean = future_returns.dropna()
    if clean.empty:
        raise ValueError("future_returns must contain at least one non-null value")

    lower = float(clean.quantile(lower_q))
    upper = float(clean.quantile(upper_q))
    return lower, upper


def label_ternary_returns(
    future_returns: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.Series:
    """Map future returns to down/flat/up labels using fixed thresholds."""
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be less than upper_threshold")

    labels = pd.Series(index=future_returns.index, dtype="object")
    labels.loc[future_returns <= lower_threshold] = "down"
    labels.loc[(future_returns > lower_threshold) & (future_returns < upper_threshold)] = "flat"
    labels.loc[future_returns >= upper_threshold] = "up"
    return labels


def build_target_frame(
    prices: pd.Series,
    primary_horizon: int = 5,
    benchmark_horizon: int = 1,
    quantiles: Iterable[float] = (1.0 / 3.0, 2.0 / 3.0),
    fit_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Build the first-pass target frame for SPY classification work."""
    primary_return = compute_future_return(prices, primary_horizon)
    benchmark_return = compute_future_return(prices, benchmark_horizon)

    fit_sample = primary_return if fit_index is None else primary_return.loc[fit_index]
    lower_threshold, upper_threshold = fit_ternary_quantile_thresholds(
        fit_sample,
        quantiles=quantiles,
    )

    frame = pd.DataFrame(
        {
            f"target_ret_{primary_horizon}d": primary_return,
            f"target_ret_{benchmark_horizon}d": benchmark_return,
            f"target_label_{primary_horizon}d": label_ternary_returns(
                primary_return,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            ),
        }
    )
    return frame, (lower_threshold, upper_threshold)
