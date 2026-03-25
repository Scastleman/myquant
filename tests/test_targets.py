from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.features.targets import (
    build_target_frame,
    compute_future_return,
    fit_ternary_quantile_thresholds,
)


class TargetTests(unittest.TestCase):
    def test_compute_future_return_uses_forward_price(self) -> None:
        prices = pd.Series([100.0, 102.0, 101.0], index=pd.RangeIndex(3))
        result = compute_future_return(prices, horizon=1)

        self.assertAlmostEqual(result.iloc[0], 0.02)
        self.assertAlmostEqual(result.iloc[1], (101.0 / 102.0) - 1.0)
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_fit_ternary_quantile_thresholds_orders_bounds(self) -> None:
        future_returns = pd.Series([-0.03, -0.01, 0.00, 0.02, 0.05])
        lower, upper = fit_ternary_quantile_thresholds(future_returns)

        self.assertLess(lower, upper)

    def test_build_target_frame_assigns_expected_labels(self) -> None:
        prices = pd.Series(
            [100.0, 101.0, 100.0, 103.0, 104.0, 105.0],
            index=pd.RangeIndex(6),
        )
        frame, thresholds = build_target_frame(
            prices,
            primary_horizon=1,
            benchmark_horizon=1,
            fit_index=pd.RangeIndex(5),
        )

        self.assertTrue({"target_ret_1d", "target_label_1d"} <= set(frame.columns))
        self.assertLess(thresholds[0], thresholds[1])
        self.assertIn(frame.loc[0, "target_label_1d"], {"down", "flat", "up"})


if __name__ == "__main__":
    unittest.main()
