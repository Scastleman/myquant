from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.training.evaluation import (  # noqa: E402
    directional_hit_rate,
    evaluate_classifier_predictions,
    multiclass_brier_score,
)


class TrainingEvaluationTests(unittest.TestCase):
    def test_multiclass_brier_score_is_zero_for_perfect_probabilities(self) -> None:
        y_true = np.array(["down", "flat", "up"])
        classes = np.array(["down", "flat", "up"])
        y_proba = np.eye(3)

        score = multiclass_brier_score(y_true, y_proba, classes)

        self.assertEqual(score, 0.0)

    def test_directional_hit_rate_ignores_flat_truth_rows(self) -> None:
        y_true = np.array(["up", "down", "flat", "up"])
        y_pred = np.array(["up", "flat", "down", "down"])

        score = directional_hit_rate(y_true, y_pred)

        self.assertAlmostEqual(score, 1.0 / 3.0)

    def test_evaluate_classifier_predictions_returns_expected_keys(self) -> None:
        y_true = np.array(["down", "flat", "up"])
        y_pred = np.array(["down", "flat", "up"])
        y_proba = np.eye(3)
        classes = np.array(["down", "flat", "up"])

        metrics = evaluate_classifier_predictions(y_true, y_pred, y_proba, classes)

        self.assertTrue(
            {
                "accuracy",
                "balanced_accuracy",
                "log_loss",
                "brier_score",
                "directional_hit_rate",
                "ece",
            }
            <= set(metrics)
        )


if __name__ == "__main__":
    unittest.main()
