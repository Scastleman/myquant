from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.planning.sequence_budget import (  # noqa: E402
    AssetGroupSpec,
    MovingAverageGroupSpec,
    ScalingPlan,
    ScenarioSpec,
    TargetSpec,
    TimeframeSpec,
    estimate_windows_per_series,
    summarize_scenario,
)


class SequenceBudgetTests(unittest.TestCase):
    def test_estimate_windows_per_series_respects_stride(self) -> None:
        count = estimate_windows_per_series(
            total_bars=1_000,
            lookback_bars=100,
            max_target_horizon_bars=20,
            stride_bars=5,
        )

        self.assertEqual(count, 177)

    def test_summarize_scenario_separates_unique_sequences_from_labeled_examples(self) -> None:
        plan = ScalingPlan(
            asset_groups=(AssetGroupSpec(name="core", asset_count=10),),
            timeframes=(TimeframeSpec(name="1min", bars_per_year=1_000.0),),
            targets=(
                TargetSpec(name="next_return", horizon_bars=1),
                TargetSpec(name="drawdown_20", horizon_bars=20),
            ),
            moving_average_groups=(
                MovingAverageGroupSpec(name="trend", windows=(9, 21), timeframes=("1min", "daily")),
            ),
            scenarios=(
                ScenarioSpec(
                    name="core_intraday",
                    asset_groups=("core",),
                    timeframes=("1min",),
                    targets=("next_return", "drawdown_20"),
                    moving_average_groups=("trend",),
                    years=2.0,
                    lookback_bars=100,
                    stride_bars=10,
                ),
            ),
        )

        summary = summarize_scenario(plan, "core_intraday")

        self.assertEqual(summary["asset_count"], 10)
        self.assertEqual(summary["target_count"], 2)
        self.assertEqual(summary["moving_average_feature_count"], 4)
        self.assertGreater(summary["labeled_examples"], summary["unique_input_sequences"])


if __name__ == "__main__":
    unittest.main()
