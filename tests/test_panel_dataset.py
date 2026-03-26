from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.config import load_project_config  # noqa: E402
from myquant.data.panel_dataset import build_panel_dataset  # noqa: E402


class PanelDatasetTests(unittest.TestCase):
    def test_build_panel_dataset_creates_rows_for_each_target_ticker(self) -> None:
        config = load_project_config("configs/project.toml")
        dates = pd.bdate_range("2020-01-01", periods=320)

        records: list[dict] = []
        for ticker_index, ticker in enumerate(config.universe.tickers):
            base_price = 50.0 + ticker_index
            for row_index, date in enumerate(dates):
                price = base_price + row_index * (0.05 + ticker_index * 0.001)
                records.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "adj_close": price,
                    }
                )

        raw_prices = pd.DataFrame.from_records(records)
        panel = build_panel_dataset(raw_prices, config)

        self.assertFalse(panel.empty)
        self.assertEqual(
            set(panel["target_ticker"].unique()),
            set(config.panel_training.target_tickers),
        )
        self.assertIn("is_focus_ticker", panel.columns)
        self.assertIn("target_ticker_SPY", panel.columns)
        self.assertEqual(panel["is_focus_ticker"].max(), 1)
        self.assertEqual(panel["split"].isin({"train", "validation", "test"}).all(), True)


if __name__ == "__main__":
    unittest.main()
