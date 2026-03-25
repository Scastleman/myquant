from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_vix_event_flags(
    frame: pd.DataFrame,
    close_column: str = "adj_close",
    thresholds: Iterable[float] = (0.10, 0.20),
) -> pd.DataFrame:
    """
    Add close-to-close VIX event flags using only information known by the close.

    The function expects a single-series VIX frame indexed by date.
    """
    if close_column not in frame:
        raise KeyError(f"'{close_column}' column not found in frame")

    result = frame.copy()
    pct_change = result[close_column].pct_change()
    result["vix_ret_1d"] = pct_change

    for threshold in thresholds:
        pct_label = int(round(threshold * 100))
        tolerance = 1e-12
        result[f"vix_up_{pct_label}pct_flag"] = (pct_change >= (threshold - tolerance)).astype("int8")
        result[f"vix_down_{pct_label}pct_flag"] = (pct_change <= (-threshold + tolerance)).astype("int8")
        result[f"vix_abs_{pct_label}pct_flag"] = (
            pct_change.abs() >= (threshold - tolerance)
        ).astype("int8")

    return result
