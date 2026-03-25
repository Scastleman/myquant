from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.features.vix_events import add_vix_event_flags


class VixEventTests(unittest.TestCase):
    def test_add_vix_event_flags_marks_10_and_20_pct_moves(self) -> None:
        frame = pd.DataFrame(
            {
                "adj_close": [20.0, 24.0, 18.0, 19.0],
            },
            index=pd.RangeIndex(4),
        )

        result = add_vix_event_flags(frame)

        self.assertEqual(result.loc[1, "vix_up_10pct_flag"], 1)
        self.assertEqual(result.loc[1, "vix_up_20pct_flag"], 1)
        self.assertEqual(result.loc[2, "vix_down_10pct_flag"], 1)
        self.assertEqual(result.loc[2, "vix_down_20pct_flag"], 1)
        self.assertEqual(result.loc[3, "vix_abs_20pct_flag"], 0)


if __name__ == "__main__":
    unittest.main()
