from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

@dataclass(frozen=True)
class RatioSpec:
    name: str
    numerator: str
    denominator: str


@dataclass(frozen=True)
class ProjectSection:
    name: str
    timezone: str
    python_version: str
    data_start_date: str


@dataclass(frozen=True)
class DataSection:
    provider: str
    storage_format: str
    price_field: str


@dataclass(frozen=True)
class StorageSection:
    intraday_root: str
    default_input_timezone: str
    partition_timezone: str


@dataclass(frozen=True)
class TargetSection:
    primary_horizon_days: int
    benchmark_horizon_days: int
    label_mode: str
    train_quantiles: tuple[float, float]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class EventSection:
    vix_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationSection:
    metrics: tuple[str, ...]
    event_slices: tuple[str, ...]


@dataclass(frozen=True)
class PanelTrainingSection:
    target_tickers: tuple[str, ...]
    evaluation_focus_ticker: str


@dataclass(frozen=True)
class UniverseSection:
    tickers: tuple[str, ...]
    ratios: tuple[RatioSpec, ...]


@dataclass(frozen=True)
class ProjectConfig:
    project: ProjectSection
    data: DataSection
    storage: StorageSection
    targets: TargetSection
    events: EventSection
    evaluation: EvaluationSection
    panel_training: PanelTrainingSection
    universe: UniverseSection


def _as_tuple(values: Any) -> tuple[Any, ...]:
    return tuple(values)


def _build_config(raw_config: dict[str, Any]) -> ProjectConfig:
    universe = raw_config["universe"]
    ratio_specs = tuple(RatioSpec(**item) for item in universe.get("ratios", []))
    raw_panel_training = raw_config.get("panel_training")
    default_panel_tickers = tuple(
        ticker for ticker in universe["tickers"] if not str(ticker).startswith("^")
    )
    panel_training = PanelTrainingSection(
        target_tickers=_as_tuple(
            (
                raw_panel_training.get("target_tickers", default_panel_tickers)
                if raw_panel_training is not None
                else default_panel_tickers
            )
        ),
        evaluation_focus_ticker=(
            raw_panel_training.get("evaluation_focus_ticker", "SPY")
            if raw_panel_training is not None
            else "SPY"
        ),
    )
    return ProjectConfig(
        project=ProjectSection(**raw_config["project"]),
        data=DataSection(**raw_config["data"]),
        storage=StorageSection(**raw_config["storage"]),
        targets=TargetSection(
            primary_horizon_days=raw_config["targets"]["primary_horizon_days"],
            benchmark_horizon_days=raw_config["targets"]["benchmark_horizon_days"],
            label_mode=raw_config["targets"]["label_mode"],
            train_quantiles=_as_tuple(raw_config["targets"]["train_quantiles"]),
            labels=_as_tuple(raw_config["targets"]["labels"]),
        ),
        events=EventSection(
            vix_thresholds=_as_tuple(raw_config["events"]["vix_thresholds"]),
        ),
        evaluation=EvaluationSection(
            metrics=_as_tuple(raw_config["evaluation"]["metrics"]),
            event_slices=_as_tuple(raw_config["evaluation"]["event_slices"]),
        ),
        panel_training=panel_training,
        universe=UniverseSection(
            tickers=_as_tuple(universe["tickers"]),
            ratios=ratio_specs,
        ),
    )


def load_project_config(path: str | Path = "configs/project.toml") -> ProjectConfig:
    """Load the repo's main TOML config into a validated immutable dataclass tree."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw_config: dict[str, Any] = tomllib.load(handle)
    return _build_config(raw_config)
