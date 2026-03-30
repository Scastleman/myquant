from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.data.large_universe_config import (  # noqa: E402
    LargeUniverseConfig,
    LargeUniverseDatasetSection,
    LargeUniverseDownloadSection,
    LargeUniverseUniverseSection,
)
from myquant.data.large_universe_dataset import build_large_universe_datasets  # noqa: E402
from myquant.data.universe import normalize_yahoo_equity_ticker  # noqa: E402


class LargeUniverseDatasetTests(unittest.TestCase):
    def test_normalize_yahoo_equity_ticker_replaces_dots(self) -> None:
        self.assertEqual(normalize_yahoo_equity_ticker("brk.b"), "BRK-B")
        self.assertEqual(normalize_yahoo_equity_ticker(" bf.b "), "BF-B")

    def test_build_large_universe_datasets_creates_relative_and_breadth_features(self) -> None:
        dates = pd.bdate_range("2018-01-01", periods=420)
        context_tickers = (
            "SPY",
            "QQQ",
            "IWM",
            "RSP",
            "XLY",
            "XLP",
            "XLK",
            "XLU",
            "XLF",
            "SMH",
            "HYG",
            "IEF",
            "TLT",
            "GLD",
            "UUP",
            "^VIX",
        )
        equity_tickers = ("AAA", "BBB", "CCC")

        records: list[dict[str, object]] = []
        steps = np.arange(len(dates), dtype=float)

        for ticker_index, ticker in enumerate(context_tickers):
            if ticker == "^VIX":
                prices = 18.0 + 1.8 * np.sin(steps / 7.0) + ((steps % 45) == 0) * 4.0
            else:
                prices = 90.0 + ticker_index * 3.0 + steps * (0.08 + ticker_index * 0.001)
                prices = prices + 0.6 * np.sin(steps / (15.0 + ticker_index))
            for date, price in zip(dates, prices, strict=False):
                records.append({"date": date, "ticker": ticker, "adj_close": float(price)})

        for ticker_index, ticker in enumerate(equity_tickers):
            prices = 40.0 + ticker_index * 9.0 + steps * (0.12 + ticker_index * 0.015)
            prices = prices + 1.25 * np.sin(steps / (11.0 + ticker_index))
            for date, price in zip(dates, prices, strict=False):
                records.append({"date": date, "ticker": ticker, "adj_close": float(price)})

        raw_prices = pd.DataFrame.from_records(records)
        membership = pd.DataFrame(
            {
                "ticker": [*equity_tickers, *context_tickers],
                "asset_type": ["equity"] * len(equity_tickers) + ["context"] * len(context_tickers),
                "gics_sector": [
                    "Information Technology",
                    "Financials",
                    "Health Care",
                    *(["context"] * len(context_tickers)),
                ],
                "has_price_data": [True] * (len(equity_tickers) + len(context_tickers)),
            }
        )
        config = LargeUniverseConfig(
            download=LargeUniverseDownloadSection(
                start_date="2018-01-01",
                end_date=None,
                wikipedia_url="https://example.com",
                batch_size=10,
                max_retries=1,
                retry_sleep_seconds=0.0,
                stock_limit=None,
            ),
            dataset=LargeUniverseDatasetSection(
                min_history_days=300,
                min_breadth_stock_count=3,
                primary_horizon_days=5,
                benchmark_horizon_days=1,
                train_quantiles=(1.0 / 3.0, 2.0 / 3.0),
                vix_thresholds=(0.10, 0.20),
            ),
            universe=LargeUniverseUniverseSection(context_tickers=context_tickers),
        )

        panel, spy_dataset = build_large_universe_datasets(raw_prices, membership, config)

        self.assertFalse(panel.empty)
        self.assertFalse(spy_dataset.empty)
        self.assertEqual(set(panel["target_ticker"].unique()), set(equity_tickers))
        self.assertTrue(panel["split"].isin({"train", "validation", "test"}).all())
        self.assertTrue(spy_dataset["split"].isin({"train", "validation", "test"}).all())
        self.assertIn("relative_spy__ret_diff_5d", panel.columns)
        self.assertIn("relative_spy__log_price_ratio", panel.columns)
        self.assertIn("breadth__pct_outperform_spy_5d", panel.columns)
        self.assertIn("sector_information_technology", panel.columns)
        self.assertIn("vix_abs_10pct_flag", panel.columns)
        self.assertTrue(panel["target_label_5d"].isin({"down", "flat", "up"}).all())
        self.assertTrue(spy_dataset["target_label_5d"].isin({"down", "flat", "up"}).all())
        self.assertEqual(sorted(panel["deoverlap_group_5d"].unique().tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(spy_dataset["deoverlap_group_5d"].unique().tolist()), [0, 1, 2, 3, 4])

        numeric_feature_columns = [
            column
            for column in panel.columns
            if column not in {"date", "split", "target_ticker"}
            and not column.startswith("target_")
            and pd.api.types.is_numeric_dtype(panel[column])
        ]
        self.assertFalse(panel.loc[:, numeric_feature_columns].isna().any().any())


if __name__ == "__main__":
    unittest.main()
