from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCALING_PLAN_PATH = REPO_ROOT / "configs" / "scaling_plan.toml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "sequence_budgets"


@dataclass(frozen=True)
class AssetGroupSpec:
    name: str
    asset_count: int
    description: str = ""


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    bars_per_year: float
    description: str = ""


@dataclass(frozen=True)
class TargetSpec:
    name: str
    horizon_bars: int
    description: str = ""


@dataclass(frozen=True)
class MovingAverageGroupSpec:
    name: str
    windows: tuple[int, ...]
    timeframes: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    asset_groups: tuple[str, ...]
    timeframes: tuple[str, ...]
    targets: tuple[str, ...]
    moving_average_groups: tuple[str, ...]
    years: float
    lookback_bars: int
    stride_bars: int
    description: str = ""


@dataclass(frozen=True)
class ScalingPlan:
    asset_groups: tuple[AssetGroupSpec, ...]
    timeframes: tuple[TimeframeSpec, ...]
    targets: tuple[TargetSpec, ...]
    moving_average_groups: tuple[MovingAverageGroupSpec, ...]
    scenarios: tuple[ScenarioSpec, ...]


def _as_tuple(values) -> tuple:
    return tuple(values)


def load_scaling_plan(path: str | Path = DEFAULT_SCALING_PLAN_PATH) -> ScalingPlan:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    defaults = raw_config.get("defaults", {})
    default_years = float(defaults.get("years", 5.0))
    default_lookback = int(defaults.get("lookback_bars", 256))
    default_stride = int(defaults.get("stride_bars", 1))

    return ScalingPlan(
        asset_groups=tuple(AssetGroupSpec(**item) for item in raw_config.get("asset_groups", [])),
        timeframes=tuple(TimeframeSpec(**item) for item in raw_config.get("timeframes", [])),
        targets=tuple(TargetSpec(**item) for item in raw_config.get("targets", [])),
        moving_average_groups=tuple(
            MovingAverageGroupSpec(
                name=item["name"],
                windows=_as_tuple(item["windows"]),
                timeframes=_as_tuple(item["timeframes"]),
                description=item.get("description", ""),
            )
            for item in raw_config.get("moving_average_groups", [])
        ),
        scenarios=tuple(
            ScenarioSpec(
                name=item["name"],
                asset_groups=_as_tuple(item["asset_groups"]),
                timeframes=_as_tuple(item["timeframes"]),
                targets=_as_tuple(item["targets"]),
                moving_average_groups=_as_tuple(item.get("moving_average_groups", [])),
                years=float(item.get("years", default_years)),
                lookback_bars=int(item.get("lookback_bars", default_lookback)),
                stride_bars=int(item.get("stride_bars", default_stride)),
                description=item.get("description", ""),
            )
            for item in raw_config.get("scenarios", [])
        ),
    )


def _index_by_name(specs) -> dict[str, object]:
    return {spec.name: spec for spec in specs}


def estimate_windows_per_series(
    *,
    total_bars: float,
    lookback_bars: int,
    max_target_horizon_bars: int,
    stride_bars: int,
) -> int:
    available = math.floor(total_bars) - lookback_bars - max_target_horizon_bars
    if available < 0:
        return 0
    return (available // stride_bars) + 1


def summarize_scenario(plan: ScalingPlan, scenario_name: str) -> dict:
    scenario_map = _index_by_name(plan.scenarios)
    asset_group_map = _index_by_name(plan.asset_groups)
    timeframe_map = _index_by_name(plan.timeframes)
    target_map = _index_by_name(plan.targets)
    moving_average_map = _index_by_name(plan.moving_average_groups)

    scenario = scenario_map[scenario_name]
    asset_count = sum(asset_group_map[name].asset_count for name in scenario.asset_groups)
    max_target_horizon = max(target_map[name].horizon_bars for name in scenario.targets)
    target_count = len(scenario.targets)
    moving_average_feature_count = sum(
        len(moving_average_map[name].windows) * len(moving_average_map[name].timeframes)
        for name in scenario.moving_average_groups
    )

    timeframe_rows: list[dict] = []
    total_unique_sequences = 0
    total_labeled_examples = 0

    for timeframe_name in scenario.timeframes:
        timeframe = timeframe_map[timeframe_name]
        total_bars = scenario.years * timeframe.bars_per_year
        windows_per_asset = estimate_windows_per_series(
            total_bars=total_bars,
            lookback_bars=scenario.lookback_bars,
            max_target_horizon_bars=max_target_horizon,
            stride_bars=scenario.stride_bars,
        )
        unique_sequences = windows_per_asset * asset_count
        labeled_examples = unique_sequences * target_count
        total_unique_sequences += unique_sequences
        total_labeled_examples += labeled_examples
        timeframe_rows.append(
            {
                "timeframe": timeframe_name,
                "bars_per_year": timeframe.bars_per_year,
                "total_bars": math.floor(total_bars),
                "windows_per_asset": windows_per_asset,
                "unique_sequences": unique_sequences,
                "labeled_examples": labeled_examples,
            }
        )

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "years": scenario.years,
        "lookback_bars": scenario.lookback_bars,
        "stride_bars": scenario.stride_bars,
        "asset_count": asset_count,
        "asset_groups": list(scenario.asset_groups),
        "timeframes": list(scenario.timeframes),
        "targets": list(scenario.targets),
        "target_count": target_count,
        "max_target_horizon_bars": max_target_horizon,
        "moving_average_groups": list(scenario.moving_average_groups),
        "moving_average_feature_count": moving_average_feature_count,
        "unique_input_sequences": total_unique_sequences,
        "labeled_examples": total_labeled_examples,
        "timeframe_breakdown": timeframe_rows,
    }


def save_report(report: dict) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = ARTIFACTS_DIR / f"sequence-budget-{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate sequence counts for multi-asset, multi-timeframe scenarios.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SCALING_PLAN_PATH),
        help="Path to the scaling TOML config.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario name. If omitted, all configured scenarios are summarized.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    plan = load_scaling_plan(args.config)
    scenario_names = [args.scenario] if args.scenario else [scenario.name for scenario in plan.scenarios]
    summaries = [summarize_scenario(plan, scenario_name) for scenario_name in scenario_names]

    for summary in summaries:
        print(
            f"{summary['scenario']}: assets={summary['asset_count']} "
            f"timeframes={len(summary['timeframes'])} targets={summary['target_count']} "
            f"unique_sequences={summary['unique_input_sequences']:,} "
            f"labeled_examples={summary['labeled_examples']:,}",
        )
        for row in summary["timeframe_breakdown"]:
            print(
                f"  {row['timeframe']:>5}: total_bars={row['total_bars']:,} "
                f"windows_per_asset={row['windows_per_asset']:,} "
                f"unique_sequences={row['unique_sequences']:,}",
            )
        print(
            f"  moving_average_feature_count={summary['moving_average_feature_count']:,} "
            "(features, not sequence multipliers)",
        )

    report = {"config": str(args.config), "summaries": summaries}
    report_path = save_report(report)
    print(f"Saved sequence budget report to {report_path}")


if __name__ == "__main__":
    main()
