from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_COLUMN = "target_label_5d"


def _as_cpu_tensor(
    values: np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype,
    share_memory: bool,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
    else:
        array = np.asarray(values, order="C")
        if not array.flags["C_CONTIGUOUS"] or not array.flags["WRITEABLE"]:
            array = np.array(array, copy=True, order="C")
        tensor = torch.as_tensor(array, dtype=dtype)
    if share_memory:
        tensor = tensor.contiguous()
        tensor.share_memory_()
    return tensor


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


def standardize_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    stds: pd.Series,
) -> np.ndarray:
    """Materialize a contiguous float32 feature matrix using train-fitted normalization."""
    standardized = (
        frame.loc[:, feature_columns]
        .astype(np.float32)
        .sub(means.astype(np.float32), axis=1)
        .div(stds.astype(np.float32), axis=1)
    )
    return np.ascontiguousarray(standardized.to_numpy(dtype=np.float32))


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
    window_stride: int = 1,
) -> list[SequenceIndex]:
    """Create endpoint indices for rolling windows, preserving historical context."""
    if window_stride < 1:
        raise ValueError("window_stride must be at least 1")

    indices: list[SequenceIndex] = []
    splits = frame["split"].tolist()
    if not group_columns:
        for endpoint in range(lookback - 1, len(frame), window_stride):
            split_name = splits[endpoint]
            if split_name in allowed_splits:
                indices.append(SequenceIndex(endpoint=endpoint, split=split_name))
        return indices

    grouped_positions = frame.groupby(list(group_columns), sort=False, dropna=False).indices
    for positions in grouped_positions.values():
        ordered_positions = np.sort(np.asarray(positions, dtype=np.int64))
        for endpoint in ordered_positions[lookback - 1 :: window_stride]:
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
        *,
        share_memory: bool = False,
    ) -> None:
        self.feature_columns = feature_columns
        self.features = _as_cpu_tensor(
            frame.loc[:, feature_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            share_memory=share_memory,
        )
        self.labels = _as_cpu_tensor(
            frame[TARGET_COLUMN].map(label_to_index).to_numpy(dtype=np.int64),
            dtype=torch.long,
            share_memory=share_memory,
        )
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
        return x, y


class MultiTaskRollingWindowDataset(Dataset):
    """Rolling window dataset that returns one label per configured task."""

    def __init__(
        self,
        frame: pd.DataFrame | None,
        feature_columns: list[str] | None,
        task_target_columns: dict[str, str] | None,
        task_label_to_index: dict[str, dict[str, int]] | None,
        sequence_indices: list[SequenceIndex],
        lookback: int,
        *,
        feature_matrix: np.ndarray | torch.Tensor | None = None,
        label_arrays: dict[str, np.ndarray | torch.Tensor] | None = None,
        feature_matrix_path: str | Path | None = None,
        label_array_paths: dict[str, str | Path] | None = None,
        share_memory: bool = False,
    ) -> None:
        self._feature_memmap_path = str(feature_matrix_path) if feature_matrix_path is not None else None
        self._feature_memmap: np.memmap | None = None
        self._label_memmap_paths = {
            task_name: str(path)
            for task_name, path in (label_array_paths or {}).items()
        }
        self._label_memmaps: dict[str, np.memmap] = {}

        if self._feature_memmap_path is not None:
            self.feature_columns = feature_columns or []
            self.features: torch.Tensor | None = None
        elif feature_matrix is None:
            if frame is None or feature_columns is None:
                raise ValueError("frame and feature_columns are required when feature_matrix is omitted")
            self.feature_columns = feature_columns
            self.features = _as_cpu_tensor(
                frame.loc[:, feature_columns].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                share_memory=share_memory,
            )
        else:
            self.feature_columns = feature_columns or []
            self.features = _as_cpu_tensor(
                feature_matrix,
                dtype=torch.float32,
                share_memory=share_memory,
            )

        if self._label_memmap_paths:
            self.labels: dict[str, torch.Tensor] = {}
            self.task_names = tuple(self._label_memmap_paths)
        elif label_arrays is None:
            if frame is None or task_target_columns is None or task_label_to_index is None:
                raise ValueError("frame, task_target_columns, and task_label_to_index are required when label_arrays is omitted")
            self.labels = {
                task_name: _as_cpu_tensor(
                    frame[target_column].map(label_to_index).to_numpy(dtype=np.int64),
                    dtype=torch.long,
                    share_memory=share_memory,
                )
                for task_name, target_column in task_target_columns.items()
                for label_to_index in [task_label_to_index[task_name]]
            }
            self.task_names = tuple(self.labels)
        else:
            self.labels = {
                task_name: _as_cpu_tensor(
                    values,
                    dtype=torch.long,
                    share_memory=share_memory,
                )
                for task_name, values in label_arrays.items()
            }
            self.task_names = tuple(self.labels)
        self.sequence_indices = sequence_indices
        self.lookback = lookback

    def _feature_source(self) -> torch.Tensor | np.memmap:
        if self.features is not None:
            return self.features
        if self._feature_memmap is None:
            if self._feature_memmap_path is None:
                raise RuntimeError("Feature memmap path is not configured.")
            self._feature_memmap = np.load(self._feature_memmap_path, mmap_mode="r")
        return self._feature_memmap

    def _label_source(self, task_name: str) -> torch.Tensor | np.memmap:
        if task_name in self.labels:
            return self.labels[task_name]
        if task_name not in self._label_memmaps:
            if task_name not in self._label_memmap_paths:
                raise RuntimeError(f"Missing memmap path for task {task_name}.")
            self._label_memmaps[task_name] = np.load(self._label_memmap_paths[task_name], mmap_mode="r")
        return self._label_memmaps[task_name]

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sequence_index = self.sequence_indices[idx]
        endpoint = sequence_index.endpoint
        start = endpoint - self.lookback + 1
        features = self._feature_source()
        if isinstance(features, torch.Tensor):
            x = features[start : endpoint + 1]
        else:
            x = torch.as_tensor(
                np.array(features[start : endpoint + 1], copy=True, order="C"),
                dtype=torch.float32,
            )
        targets = {
            task_name: (
                label_values[endpoint]
                if isinstance(label_values := self._label_source(task_name), torch.Tensor)
                else torch.tensor(int(label_values[endpoint]), dtype=torch.long)
            )
            for task_name in self.task_names
        }
        return x, targets
