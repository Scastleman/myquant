from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_COLUMN = "target_label_5d"


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric non-target feature columns shared by baseline and transformer training."""
    excluded = {"date", "split"}
    excluded_prefixes = ("target_",)

    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(column)
    return feature_columns


def fit_standardization_stats(train_frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    """Fit mean/std normalization on the train split only."""
    means = train_frame.loc[:, feature_columns].mean()
    stds = train_frame.loc[:, feature_columns].std().replace(0.0, 1.0).fillna(1.0)
    return means, stds


def apply_standardization(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Apply train-fitted standardization to the full frame."""
    normalized = frame.copy()
    standardized = (
        normalized.loc[:, feature_columns]
        .astype(np.float32)
        .sub(means.astype(np.float32), axis=1)
        .div(stds.astype(np.float32), axis=1)
        .astype(np.float32)
    )
    normalized = normalized.drop(columns=feature_columns)
    normalized = pd.concat([normalized, standardized], axis=1)
    return normalized


def build_label_mapping(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    forward = {label: idx for idx, label in enumerate(labels)}
    reverse = {idx: label for label, idx in forward.items()}
    return forward, reverse


@dataclass(frozen=True)
class SequenceIndex:
    endpoint: int
    split: str


def build_sequence_indices(
    frame: pd.DataFrame,
    lookback: int,
    allowed_splits: tuple[str, ...],
    group_columns: tuple[str, ...] | None = None,
) -> list[SequenceIndex]:
    """Create endpoint indices for rolling windows, preserving historical context."""
    indices: list[SequenceIndex] = []
    splits = frame["split"].tolist()
    if not group_columns:
        for endpoint in range(lookback - 1, len(frame)):
            split_name = splits[endpoint]
            if split_name in allowed_splits:
                indices.append(SequenceIndex(endpoint=endpoint, split=split_name))
        return indices

    grouped_positions = frame.groupby(list(group_columns), sort=False, dropna=False).indices
    for positions in grouped_positions.values():
        ordered_positions = np.sort(np.asarray(positions, dtype=np.int64))
        for endpoint in ordered_positions[lookback - 1 :]:
            split_name = splits[int(endpoint)]
            if split_name in allowed_splits:
                indices.append(SequenceIndex(endpoint=int(endpoint), split=split_name))
    return indices


class RollingWindowDataset(Dataset):
    """Rolling window dataset over the processed date-level feature table."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        label_to_index: dict[str, int],
        sequence_indices: list[SequenceIndex],
        lookback: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.features = np.ascontiguousarray(frame.loc[:, feature_columns].to_numpy(dtype=np.float32))
        self.labels = frame[TARGET_COLUMN].map(label_to_index).to_numpy(dtype=np.int64)
        self.sequence_indices = sequence_indices
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_index = self.sequence_indices[idx]
        endpoint = sequence_index.endpoint
        start = endpoint - self.lookback + 1
        x = self.features[start : endpoint + 1]
        y = self.labels[endpoint]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
