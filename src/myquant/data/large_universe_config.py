from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class LargeUniverseDownloadSection:
    start_date: str
    end_date: str | None
    wikipedia_url: str
    batch_size: int
    max_retries: int
    retry_sleep_seconds: float
    stock_limit: int | None


@dataclass(frozen=True)
class LargeUniverseDatasetSection:
    min_history_days: int
    min_breadth_stock_count: int
    primary_horizon_days: int
    benchmark_horizon_days: int
    train_quantiles: tuple[float, float]
    vix_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class LargeUniverseUniverseSection:
    context_tickers: tuple[str, ...]


@dataclass(frozen=True)
class LargeUniverseConfig:
    download: LargeUniverseDownloadSection
    dataset: LargeUniverseDatasetSection
    universe: LargeUniverseUniverseSection


def _as_tuple(values: Any) -> tuple[Any, ...]:
    return tuple(values)


def load_large_universe_config(path: str | Path = "configs/large_universe.toml") -> LargeUniverseConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw_config: dict[str, Any] = tomllib.load(handle)

    raw_download = raw_config["download"]
    raw_dataset = raw_config["dataset"]
    return LargeUniverseConfig(
        download=LargeUniverseDownloadSection(
            start_date=raw_download["start_date"],
            end_date=raw_download.get("end_date"),
            wikipedia_url=raw_download["wikipedia_url"],
            batch_size=int(raw_download["batch_size"]),
            max_retries=int(raw_download["max_retries"]),
            retry_sleep_seconds=float(raw_download["retry_sleep_seconds"]),
            stock_limit=(
                int(raw_download["stock_limit"])
                if raw_download.get("stock_limit") is not None
                else None
            ),
        ),
        dataset=LargeUniverseDatasetSection(
            min_history_days=int(raw_dataset["min_history_days"]),
            min_breadth_stock_count=int(raw_dataset["min_breadth_stock_count"]),
            primary_horizon_days=int(raw_dataset["primary_horizon_days"]),
            benchmark_horizon_days=int(raw_dataset["benchmark_horizon_days"]),
            train_quantiles=_as_tuple(raw_dataset["train_quantiles"]),
            vix_thresholds=_as_tuple(raw_dataset["vix_thresholds"]),
        ),
        universe=LargeUniverseUniverseSection(
            context_tickers=_as_tuple(raw_config["universe"]["context_tickers"]),
        ),
    )
