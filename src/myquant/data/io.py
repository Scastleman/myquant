from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
LAKE_DIR = DATA_DIR / "lake"
RAW_PRICES_PATH = DATA_DIR / "raw" / "prices.parquet"
LARGE_UNIVERSE_MEMBERSHIP_PATH = DATA_DIR / "raw" / "large_universe_membership.parquet"
LARGE_UNIVERSE_RAW_PRICES_PATH = DATA_DIR / "raw" / "large_universe_prices.parquet"
PROCESSED_DATASET_PATH = DATA_DIR / "processed" / "phase1_dataset.parquet"
PANEL_DATASET_PATH = DATA_DIR / "processed" / "panel_dataset.parquet"
LARGE_UNIVERSE_PANEL_DATASET_PATH = DATA_DIR / "processed" / "large_universe_panel_dataset.parquet"
SPY_BREADTH_DATASET_PATH = DATA_DIR / "processed" / "spy_breadth_dataset.parquet"
INTRADAY_BAR_STORE_ROOT = LAKE_DIR / "bars"


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if it does not already exist."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_parquet(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a DataFrame to Parquet with a clear error if parquet support is missing."""
    target = ensure_parent_dir(path)
    try:
        frame.to_parquet(target, index=index)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is unavailable. Install 'pyarrow' in the project environment.",
        ) from exc
    return target


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    target = Path(path)
    try:
        return pd.read_parquet(target)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is unavailable. Install 'pyarrow' in the project environment.",
        ) from exc
