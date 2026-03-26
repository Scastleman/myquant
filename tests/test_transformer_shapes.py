from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myquant.models import PatchTransformerClassifier  # noqa: E402
from myquant.training.sequence_data import (  # noqa: E402
    RollingWindowDataset,
    build_label_mapping,
    build_sequence_indices,
    get_feature_columns,
)


class TransformerShapeTests(unittest.TestCase):
    def test_patch_transformer_forward_shape(self) -> None:
        model = PatchTransformerClassifier(
            feature_dim=8,
            sequence_length=20,
            num_classes=3,
            patch_length=5,
            patch_stride=5,
            d_model=32,
            n_heads=4,
            num_layers=2,
        )
        x = torch.randn(4, 20, 8)
        logits = model(x)

        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_rolling_window_dataset_returns_expected_shape(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="D"),
                "feature_a": np.arange(8, dtype=float),
                "feature_b": np.arange(8, dtype=float) * 2,
                "target_label_5d": ["down", "flat", "up", "down", "flat", "up", "down", "flat"],
                "split": ["train", "train", "train", "train", "validation", "validation", "test", "test"],
            }
        )
        labels = sorted(frame["target_label_5d"].unique().tolist())
        label_to_index, _ = build_label_mapping(labels)
        feature_columns = get_feature_columns(frame)
        sequence_indices = build_sequence_indices(frame, lookback=4, allowed_splits=("validation", "test"))
        dataset = RollingWindowDataset(
            frame=frame,
            feature_columns=feature_columns,
            label_to_index=label_to_index,
            sequence_indices=sequence_indices,
            lookback=4,
        )

        x, y = dataset[0]
        self.assertEqual(tuple(x.shape), (4, 2))
        self.assertEqual(tuple(y.shape), ())

    def test_grouped_sequence_indices_do_not_cross_target_ticker_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="D"),
                "target_ticker": ["AAA"] * 4 + ["BBB"] * 4,
                "feature_a": np.arange(8, dtype=float),
                "feature_b": np.arange(8, dtype=float) * 2,
                "target_label_5d": ["down", "flat", "up", "down", "flat", "up", "down", "flat"],
                "split": [
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "train",
                    "train",
                    "validation",
                    "validation",
                ],
            }
        )
        labels = sorted(frame["target_label_5d"].unique().tolist())
        label_to_index, _ = build_label_mapping(labels)
        feature_columns = get_feature_columns(frame)
        sequence_indices = build_sequence_indices(
            frame,
            lookback=3,
            allowed_splits=("validation",),
            group_columns=("target_ticker",),
        )
        dataset = RollingWindowDataset(
            frame=frame,
            feature_columns=feature_columns,
            label_to_index=label_to_index,
            sequence_indices=sequence_indices,
            lookback=3,
        )

        self.assertEqual([item.endpoint for item in sequence_indices], [2, 3, 6, 7])
        x, _ = dataset[2]
        self.assertTrue(np.array_equal(x.numpy()[:, 0], np.array([4.0, 5.0, 6.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
