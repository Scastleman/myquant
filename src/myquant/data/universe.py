from __future__ import annotations

from io import StringIO
from typing import Final

import pandas as pd
import requests


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

SP500_WIKIPEDIA_URL: Final[str] = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_yahoo_equity_ticker(symbol: str) -> str:
    """Convert common equity ticker formats into Yahoo-compatible symbols."""
    return str(symbol).strip().upper().replace(".", "-")


def fetch_current_sp500_constituents(source_url: str = SP500_WIKIPEDIA_URL) -> pd.DataFrame:
    """Fetch the current S&P 500 constituent table from Wikipedia."""
    response = requests.get(
        source_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; myquant/0.1)"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("No tables were found when fetching the S&P 500 constituent list.")

    constituents = tables[0].copy()
    required_columns = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    missing = required_columns - set(constituents.columns)
    if missing:
        raise RuntimeError(f"S&P 500 constituent table is missing expected columns: {sorted(missing)}")

    constituents = constituents.loc[:, ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    constituents = constituents.rename(
        columns={
            "Symbol": "symbol",
            "Security": "security",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
        }
    )
    constituents["ticker"] = constituents["symbol"].map(normalize_yahoo_equity_ticker)
    constituents = constituents.sort_values("ticker").reset_index(drop=True)
    return constituents
