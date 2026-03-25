from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.config.settings import RatioSpec
from myquant.features.market_features import assign_time_splits, compute_ratio_prices


class MarketFeatureTests(unittest.TestCase):
    def test_assign_time_splits_covers_all_rows_in_order(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        split = assign_time_splits(index)

        self.assertEqual(split.iloc[0], "train")
        self.assertEqual(split.iloc[-1], "test")
        self.assertIn("validation", set(split.tolist()))

    def test_compute_ratio_prices_uses_named_specs(self) -> None:
        prices = pd.DataFrame(
            {
                "QQQ": [300.0, 303.0],
                "SPY": [400.0, 404.0],
            },
            index=pd.RangeIndex(2),
        )
        ratio_specs = (RatioSpec(name="QQQ_over_SPY", numerator="QQQ", denominator="SPY"),)

        result = compute_ratio_prices(prices, ratio_specs)

        self.assertListEqual(list(result.columns), ["QQQ_over_SPY"])
        self.assertAlmostEqual(result.loc[0, "QQQ_over_SPY"], 0.75)


if __name__ == "__main__":
    unittest.main()
