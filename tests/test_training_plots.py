from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.training.plots import plot_experiment_comparison, plot_training_curves  # noqa: E402


class TrainingPlotTests(unittest.TestCase):
    def test_plot_training_curves_writes_dashboard(self) -> None:
        history = [
            {
                "epoch": 1,
                "train_loss": 1.2,
                "train_primary_loss": 1.0,
                "train_auxiliary_loss": 0.6,
                "val_loss": 1.3,
                "val_primary_loss": 1.1,
                "val_auxiliary_loss": 0.7,
                "val_balanced_accuracy": 0.4,
                "val_directional_hit_rate": 0.5,
                "learning_rate": 3e-4,
                "val_pred_share_down": 0.3,
                "val_pred_share_flat": 0.4,
                "val_pred_share_up": 0.3,
            }
        ]
        step_history = [
            {
                "global_step": 10,
                "running_loss": 1.15,
                "running_primary_loss": 1.01,
            }
        ]
        telemetry_history = [
            {
                "elapsed_seconds": 30.0,
                "gpu_utilization_pct": 78.0,
                "power_draw_w": 182.0,
                "power_limit_w": 300.0,
                "memory_used_gb": 11.4,
                "torch_reserved_gb": 10.8,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "training_progress.png"
            target = plot_training_curves(
                history,
                output_path,
                step_history=step_history,
                telemetry_history=telemetry_history,
            )

            self.assertEqual(target, output_path)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)

    def test_plot_training_curves_handles_empty_histories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "training_progress.png"
            target = plot_training_curves([], output_path)

            self.assertEqual(target, output_path)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)

    def test_plot_experiment_comparison_writes_summary_chart(self) -> None:
        rows = [
            {
                "name": "patchtst_core",
                "validation_log_loss": 1.02,
                "test_log_loss": 1.05,
                "validation_balanced_accuracy": 0.41,
                "test_balanced_accuracy": 0.40,
                "validation_directional_hit_rate": 0.47,
                "test_directional_hit_rate": 0.46,
                "model_parameter_count": 1_200_000,
                "best_epoch": 3,
            },
            {
                "name": "lighter_backbone",
                "validation_log_loss": 1.01,
                "test_log_loss": 1.04,
                "validation_balanced_accuracy": 0.42,
                "test_balanced_accuracy": 0.41,
                "validation_directional_hit_rate": 0.48,
                "test_directional_hit_rate": 0.47,
                "model_parameter_count": 900_000,
                "best_epoch": 4,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparison.png"
            target = plot_experiment_comparison(rows, output_path)

            self.assertEqual(target, output_path)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
