from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.training.run_transformer import (  # noqa: E402
    _optimizer_steps_per_epoch,
    build_warmup_cosine_scheduler,
    cleanup_runtime,
    summarize_step_history,
    summarize_telemetry_history,
)


class TransformerScheduleTests(unittest.TestCase):
    def test_optimizer_steps_per_epoch_respects_accumulation(self) -> None:
        self.assertEqual(_optimizer_steps_per_epoch(10, 4), 3)
        self.assertEqual(_optimizer_steps_per_epoch(8, 2), 4)

    def test_warmup_cosine_scheduler_warms_up_then_decays(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=10,
            warmup_steps=2,
            min_lr_ratio=0.2,
        )

        lrs: list[float] = []
        self.assertLess(float(optimizer.param_groups[0]["lr"]), 1.0)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(float(optimizer.param_groups[0]["lr"]))

        self.assertAlmostEqual(lrs[0], 1.0, places=6)
        self.assertLess(lrs[-1], lrs[2])
        self.assertGreaterEqual(lrs[-1], 0.2)
        self.assertLessEqual(max(lrs), 1.0)

    def test_cleanup_runtime_is_safe_on_cpu(self) -> None:
        cleanup_runtime(torch.device("cpu"))

    def test_summarize_telemetry_history_tracks_power_when_available(self) -> None:
        summary = summarize_telemetry_history(
            [
                {
                    "gpu_utilization_pct": 90.0,
                    "memory_used_gb": 11.0,
                    "torch_reserved_gb": 10.5,
                    "temperature_c": 64.0,
                    "graphics_clock_mhz": 2450.0,
                    "sm_clock_mhz": 2450.0,
                    "power_draw_w": 150.0,
                    "power_limit_w": 300.0,
                },
                {
                    "gpu_utilization_pct": 95.0,
                    "memory_used_gb": 12.0,
                    "torch_reserved_gb": 11.0,
                    "temperature_c": 66.0,
                    "graphics_clock_mhz": 2520.0,
                    "sm_clock_mhz": 2520.0,
                    "power_draw_w": 180.0,
                    "power_limit_w": 300.0,
                },
                {
                    "gpu_utilization_pct": 92.0,
                    "memory_used_gb": 12.5,
                    "torch_reserved_gb": 11.2,
                    "temperature_c": 65.0,
                    "graphics_clock_mhz": 2490.0,
                    "sm_clock_mhz": 2490.0,
                    "power_draw_w": float("nan"),
                    "power_limit_w": 300.0,
                },
            ]
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["power_draw_w_sample_count"], 2)
        self.assertAlmostEqual(summary["average_power_draw_w"], 165.0)
        self.assertAlmostEqual(summary["peak_power_draw_w"], 180.0)
        self.assertAlmostEqual(summary["average_power_pct_of_limit"], 55.0)
        self.assertAlmostEqual(summary["peak_power_pct_of_limit"], 60.0)
        self.assertAlmostEqual(summary["average_temperature_c"], 65.0)
        self.assertAlmostEqual(summary["peak_graphics_clock_mhz"], 2520.0)

    def test_summarize_step_history_tracks_throughput(self) -> None:
        summary = summarize_step_history(
            [
                {"samples_per_second": 420.0, "batches_per_second": 0.55},
                {"samples_per_second": 480.0, "batches_per_second": 0.62},
                {"samples_per_second": float("nan"), "batches_per_second": 0.60},
            ]
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["samples_per_second_sample_count"], 2)
        self.assertAlmostEqual(summary["average_samples_per_second"], 450.0)
        self.assertAlmostEqual(summary["peak_samples_per_second"], 480.0)
        self.assertAlmostEqual(summary["average_batches_per_second"], 0.59, places=6)


if __name__ == "__main__":
    unittest.main()
