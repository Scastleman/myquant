from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.storage.bar_store import (  # noqa: E402
    build_bar_partition_path,
    query_bar_store,
    summarize_bar_store,
    write_bar_batch,
)


class BarStoreTests(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-27 09:30:00",
                    "2026-03-27 09:31:00",
                    "2026-03-30 09:30:00",
                ],
                "ticker": ["spy", "qqq", "spy"],
                "open": [100.0, 200.0, 101.0],
                "high": [101.0, 201.0, 102.0],
                "low": [99.5, 199.5, 100.5],
                "close": [100.5, 200.5, 101.5],
                "volume": [1_000, 2_000, 1_500],
            }
        )

    def test_write_bar_batch_creates_daily_partitions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results = write_bar_batch(
                self._sample_frame(),
                source="Polygon.io",
                timeframe="1m",
                root=tmp_dir,
            )

            self.assertEqual(len(results), 2)
            expected_path = build_bar_partition_path(
                root=tmp_dir,
                source="Polygon.io",
                timeframe="1m",
                session_date="2026-03-27",
            )
            self.assertTrue(expected_path.exists())

            stored = pd.read_parquet(expected_path)
            self.assertEqual(list(stored["ticker"]), ["SPY", "QQQ"])
            self.assertNotIn("session_date", stored.columns)

    def test_query_bar_store_reads_hive_partitions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            write_bar_batch(
                self._sample_frame(),
                source="Polygon.io",
                timeframe="1m",
                root=tmp_dir,
            )

            result = query_bar_store(
                root=tmp_dir,
                tickers=["SPY"],
                timeframe="1m",
                start="2026-03-27 00:00:00",
                end="2026-03-31 00:00:00",
            )

            self.assertEqual(len(result), 2)
            self.assertEqual(set(result["source"]), {"polygon_io"})
            self.assertEqual(set(result["timeframe"]), {"1min"})

    def test_summarize_bar_store_uses_metadata_sidecars(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            write_bar_batch(
                self._sample_frame(),
                source="Polygon.io",
                timeframe="1m",
                root=tmp_dir,
            )

            summary = summarize_bar_store(tmp_dir)

            self.assertEqual(len(summary), 2)
            self.assertEqual(int(summary["row_count"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
