from __future__ import annotations

from typing import Final


PHASE1_TICKERS: Final[tuple[str, ...]] = (
    "SPY",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "QQQ",
    "IWM",
    "DIA",
    "MDY",
    "RSP",
    "SMH",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "UUP",
    "^VIX",
)

RATIO_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("QQQ_over_SPY", "QQQ", "SPY"),
    ("IWM_over_SPY", "IWM", "SPY"),
    ("RSP_over_SPY", "RSP", "SPY"),
    ("XLY_over_XLP", "XLY", "XLP"),
    ("XLK_over_XLU", "XLK", "XLU"),
    ("XLF_over_XLU", "XLF", "XLU"),
    ("SMH_over_SPY", "SMH", "SPY"),
    ("HYG_over_IEF", "HYG", "IEF"),
    ("SPY_over_TLT", "SPY", "TLT"),
    ("GLD_over_UUP", "GLD", "UUP"),
)
