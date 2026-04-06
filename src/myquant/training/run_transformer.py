from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
import gc
import math
import os
import subprocess
import time
import webbrowser

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import pynvml
except ImportError:  # pragma: no cover - optional runtime dependency
    pynvml = None

from myquant.data.io import PROCESSED_DATASET_PATH, read_parquet
from myquant.models import PatchTransformerClassifier

from .artifacts import create_run_dir, write_json
from .evaluation import evaluate_classifier_predictions
from .plots import plot_confusion_matrix, plot_training_curves
from .sequence_data import (
    MultiTaskRollingWindowDataset,
    build_label_mapping,
    build_sequence_indices,
    fit_standardization_stats,
    get_feature_columns,
    standardize_feature_matrix,
)


@dataclass(frozen=True)
class TransformerConfig:
    lookback: int = 60
    window_stride: int = 5
    patch_length: int = 5
    patch_stride: int = 5
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    regime_dim: int = 32
    dropout: float = 0.1
    attention_dropout: float = 0.0
    ff_dim: int | None = None
    norm_first: bool = True
    use_revin: bool = False
    revin_affine: bool = True
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 3e-4
    warmup_fraction: float = 0.1
    min_lr_ratio: float = 0.1
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    device: str = "auto"
    num_workers: int = 2
    loader_cache_mode: str = "auto"
    amp: bool = True
    amp_dtype: str = "auto"
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    random_state: int = 42
    primary_target_column: str = "target_label_5d"
    auxiliary_target_columns: tuple[str, ...] = ()
    auxiliary_loss_weight: float = 0.5
    focus_ticker: str = "SPY"
    open_dashboard: bool = True


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_column: str
    labels: tuple[str, ...]
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]


@dataclass(frozen=True)
class TransformerPredictionOutput:
    logits: np.ndarray
    probability_distribution: np.ndarray
    regime_state: np.ndarray
    targets: np.ndarray
    task_logits: dict[str, np.ndarray]
    task_probability_distributions: dict[str, np.ndarray]
    task_targets: dict[str, np.ndarray]
    primary_task_name: str


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(requested_device)


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def cleanup_runtime(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()


def _build_tensor_backed_arrays(
    feature_matrix: np.ndarray,
    label_arrays: dict[str, np.ndarray],
    *,
    share_memory: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature_source = (
        feature_matrix
        if feature_matrix.flags["C_CONTIGUOUS"] and feature_matrix.flags["WRITEABLE"]
        else np.array(feature_matrix, copy=True, order="C")
    )
    feature_tensor = torch.from_numpy(feature_source)
    if share_memory:
        feature_tensor = feature_tensor.contiguous()
        feature_tensor.share_memory_()

    label_tensors: dict[str, torch.Tensor] = {}
    for task_name, values in label_arrays.items():
        label_source = (
            values
            if values.flags["C_CONTIGUOUS"] and values.flags["WRITEABLE"]
            else np.array(values, copy=True, order="C")
        )
        label_tensor = torch.from_numpy(label_source)
        if share_memory:
            label_tensor = label_tensor.contiguous()
            label_tensor.share_memory_()
        label_tensors[task_name] = label_tensor
    return feature_tensor, label_tensors


_NVML_INITIALIZED = False


def _ensure_nvml_initialized() -> bool:
    global _NVML_INITIALIZED
    if pynvml is None:
        return False
    if _NVML_INITIALIZED:
        return True
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:  # type: ignore[union-attr]
        return False
    _NVML_INITIALIZED = True
    return True


@lru_cache(maxsize=8)
def _query_driver_model(device_index: int) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=driver_model.current,driver_model.pending",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None

    values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(values) != 2:
        return None, None
    current, pending = values
    current = None if current.upper() == "N/A" else current
    pending = None if pending.upper() == "N/A" else pending
    return current, pending


def _query_power_via_nvml(device_index: int) -> dict[str, float | str | bool | None]:
    if not _ensure_nvml_initialized():
        return {
            "power_draw_w": float("nan"),
            "power_limit_w": float("nan"),
            "power_source": "nvml",
            "power_supported": False,
            "power_unavailable_reason": "nvml_unavailable",
        }

    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)  # type: ignore[union-attr]
    try:
        power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # type: ignore[union-attr]
        power_supported = True
        power_reason = None
    except pynvml.NVMLError_NotSupported:  # type: ignore[union-attr]
        power_draw_w = float("nan")
        power_supported = False
        driver_model_current, _ = _query_driver_model(device_index)
        power_reason = (
            "nvml_power_not_supported_under_wddm"
            if driver_model_current == "WDDM"
            else "nvml_power_not_supported"
        )
    except pynvml.NVMLError:  # type: ignore[union-attr]
        power_draw_w = float("nan")
        power_supported = False
        power_reason = "nvml_power_query_failed"

    try:
        power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # type: ignore[union-attr]
    except pynvml.NVMLError:  # type: ignore[union-attr]
        power_limit_w = float("nan")

    return {
        "power_draw_w": power_draw_w,
        "power_limit_w": power_limit_w,
        "power_source": "nvml",
        "power_supported": power_supported,
        "power_unavailable_reason": power_reason,
    }


def query_gpu_telemetry(
    device: torch.device,
    *,
    elapsed_seconds: float,
) -> dict[str, float | str | bool | None] | None:
    if device.type != "cuda":
        return None
    device_index = device.index if device.index is not None else 0

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    raw_values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(raw_values) != 9:
        return None

    try:
        def parse_value(value: str) -> float:
            normalized = value.strip().upper()
            if normalized in {"N/A", "[N/A]"}:
                return float("nan")
            return float(value)

        gpu_utilization_pct, memory_utilization_pct, memory_used_mib, memory_total_mib, power_draw_w, power_limit_w, temperature_c, graphics_clock_mhz, sm_clock_mhz = (
            parse_value(item) for item in raw_values
        )
    except ValueError:
        return None

    power_info = _query_power_via_nvml(device_index)
    power_draw_from_nvml = float(power_info["power_draw_w"])
    power_limit_from_nvml = float(power_info["power_limit_w"])
    power_draw_final = power_draw_from_nvml if bool(np.isfinite(power_draw_from_nvml)) else power_draw_w
    power_limit_final = power_limit_from_nvml if bool(np.isfinite(power_limit_from_nvml)) else power_limit_w
    power_source = str(power_info["power_source"])
    power_supported = bool(bool(power_info["power_supported"]) or bool(np.isfinite(power_draw_w)))
    power_unavailable_reason = power_info["power_unavailable_reason"]
    driver_model_current, driver_model_pending = _query_driver_model(device_index)
    if not power_supported and power_unavailable_reason is None:
        power_unavailable_reason = "nvidia_smi_power_not_available"

    return {
        "elapsed_seconds": float(elapsed_seconds),
        "gpu_utilization_pct": gpu_utilization_pct,
        "memory_utilization_pct": memory_utilization_pct,
        "memory_used_gb": memory_used_mib / 1024.0,
        "memory_total_gb": memory_total_mib / 1024.0,
        "power_draw_w": power_draw_final,
        "power_limit_w": power_limit_final,
        "power_source": power_source,
        "power_supported": power_supported,
        "power_unavailable_reason": power_unavailable_reason,
        "driver_model_current": driver_model_current,
        "driver_model_pending": driver_model_pending,
        "temperature_c": temperature_c,
        "graphics_clock_mhz": graphics_clock_mhz,
        "sm_clock_mhz": sm_clock_mhz,
        "torch_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "torch_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "torch_max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


def summarize_telemetry_history(
    telemetry_history: list[dict[str, float | str | bool | None]],
) -> dict[str, float | int | str | bool | None]:
    def finite_values(key: str) -> list[float]:
        values = [float(item[key]) for item in telemetry_history if key in item and np.isfinite(item[key])]
        return values

    def stats(key: str, prefix: str) -> dict[str, float | int | None]:
        values = finite_values(key)
        if not values:
            return {
                f"{prefix}_sample_count": 0,
                f"average_{prefix}": None,
                f"peak_{prefix}": None,
            }
        return {
            f"{prefix}_sample_count": len(values),
            f"average_{prefix}": float(np.mean(values)),
            f"peak_{prefix}": float(np.max(values)),
        }

    summary: dict[str, float | int | str | bool | None] = {"sample_count": len(telemetry_history)}
    summary.update(stats("gpu_utilization_pct", "gpu_utilization_pct"))
    summary.update(stats("memory_used_gb", "board_vram_gb"))
    summary.update(stats("torch_reserved_gb", "torch_reserved_gb"))
    summary.update(stats("temperature_c", "temperature_c"))
    summary.update(stats("graphics_clock_mhz", "graphics_clock_mhz"))
    summary.update(stats("sm_clock_mhz", "sm_clock_mhz"))
    summary.update(stats("power_draw_w", "power_draw_w"))

    power_draw_values = finite_values("power_draw_w")
    power_limit_values = finite_values("power_limit_w")
    if power_draw_values and power_limit_values:
        limit = power_limit_values[-1]
        if np.isfinite(limit) and limit > 0:
            power_pct_values = [(value / limit) * 100.0 for value in power_draw_values]
            summary.update(
                {
                    "power_pct_of_limit_sample_count": len(power_pct_values),
                    "average_power_pct_of_limit": float(np.mean(power_pct_values)),
                    "peak_power_pct_of_limit": float(np.max(power_pct_values)),
                }
            )
        else:
            summary.update(
                {
                    "power_pct_of_limit_sample_count": 0,
                    "average_power_pct_of_limit": None,
                    "peak_power_pct_of_limit": None,
                }
            )
    else:
        summary.update(
            {
                "power_pct_of_limit_sample_count": 0,
                "average_power_pct_of_limit": None,
                "peak_power_pct_of_limit": None,
            }
        )

    power_reason_counts: dict[str, int] = {}
    power_source_counts: dict[str, int] = {}
    for item in telemetry_history:
        source = item.get("power_source")
        if isinstance(source, str):
            power_source_counts[source] = power_source_counts.get(source, 0) + 1
        reason = item.get("power_unavailable_reason")
        if isinstance(reason, str):
            power_reason_counts[reason] = power_reason_counts.get(reason, 0) + 1

    if power_source_counts:
        dominant_source = max(power_source_counts, key=power_source_counts.get)
        summary["power_source"] = dominant_source
    else:
        summary["power_source"] = None

    summary["power_supported"] = summary["power_draw_w_sample_count"] > 0
    summary["power_unavailable_reason"] = (
        max(power_reason_counts, key=power_reason_counts.get)
        if power_reason_counts
        else None
    )

    return summary


def summarize_step_history(
    step_history: list[dict[str, float]],
) -> dict[str, float | int | None]:
    def finite_values(key: str) -> list[float]:
        values = [float(item[key]) for item in step_history if key in item and np.isfinite(item[key])]
        return values

    def stats(key: str, prefix: str) -> dict[str, float | int | None]:
        values = finite_values(key)
        if not values:
            return {
                f"{prefix}_sample_count": 0,
                f"average_{prefix}": None,
                f"peak_{prefix}": None,
            }
        return {
            f"{prefix}_sample_count": len(values),
            f"average_{prefix}": float(np.mean(values)),
            f"peak_{prefix}": float(np.max(values)),
        }

    summary: dict[str, float | int | None] = {"sample_count": len(step_history)}
    summary.update(stats("samples_per_second", "samples_per_second"))
    summary.update(stats("batches_per_second", "batches_per_second"))
    return summary


def _write_live_dashboard(
    *,
    dashboard_path: Path,
    live_plot_path: Path,
    run_dir: Path,
    status: dict,
    refresh_seconds: int = 5,
) -> Path:
    latest_epoch = status.get("latest_epoch")
    latest_step = status.get("latest_step")
    total_steps = status.get("total_steps")
    best_epoch = status.get("best_epoch_so_far")
    gpu_summary = status.get("latest_gpu")
    current_loss = status.get("latest_running_loss")
    current_lr = status.get("latest_learning_rate")
    current_samples_per_second = status.get("latest_samples_per_second")
    phase = status.get("phase", "starting")

    metric_rows = [
        ("Phase", phase),
        ("Epoch", f"{latest_epoch}/{status.get('total_epochs', '?')}" if latest_epoch is not None else "waiting"),
        ("Step", f"{latest_step}/{total_steps}" if latest_step is not None and total_steps is not None else "waiting"),
        ("Best Epoch", best_epoch if best_epoch is not None else "waiting"),
        ("Latest Running Loss", f"{current_loss:.4f}" if isinstance(current_loss, float) else "waiting"),
        ("Latest LR", f"{current_lr:.2e}" if isinstance(current_lr, float) else "waiting"),
        ("Throughput", f"{current_samples_per_second:.1f} samples/s" if isinstance(current_samples_per_second, float) else "waiting"),
    ]
    if isinstance(gpu_summary, dict):
        power_value = (
            f"{gpu_summary.get('power_draw_w', float('nan')):.1f} / {gpu_summary.get('power_limit_w', float('nan')):.1f} W"
            if np.isfinite(gpu_summary.get("power_draw_w", float("nan")))
            else f"Unavailable ({gpu_summary.get('power_unavailable_reason', 'unknown')})"
        )
        metric_rows.extend(
            [
                ("GPU Util", f"{gpu_summary.get('gpu_utilization_pct', float('nan')):.0f}%"),
                ("GPU Temp", f"{gpu_summary.get('temperature_c', float('nan')):.0f} C"),
                ("Graphics Clock", f"{gpu_summary.get('graphics_clock_mhz', float('nan')):.0f} MHz"),
                ("SM Clock", f"{gpu_summary.get('sm_clock_mhz', float('nan')):.0f} MHz"),
                ("Board VRAM", f"{gpu_summary.get('memory_used_gb', float('nan')):.2f} / {gpu_summary.get('memory_total_gb', float('nan')):.2f} GB"),
                ("Power", power_value),
                ("Torch Reserved", f"{gpu_summary.get('torch_reserved_gb', float('nan')):.2f} GB"),
            ]
        )

    metrics_html = "".join(
        f"<div class='metric'><span class='label'>{escape(str(label))}</span><span class='value'>{escape(str(value))}</span></div>"
        for label, value in metric_rows
    )
    image_href = f"{live_plot_path.name}?t={int(time.time())}"
    summary_href = "summary.json"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>myquant Live Training Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ color: #94a3b8; margin: 0 0 18px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .metric {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 12px 14px; }}
    .label {{ display: block; color: #94a3b8; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .value {{ display: block; font-size: 20px; font-weight: 600; }}
    .panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 16px; }}
    img {{ width: 100%; height: auto; border-radius: 12px; display: block; }}
    .links {{ margin-top: 12px; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>myquant Live Training Dashboard</h1>
    <p>Auto-refreshing view for the current run at {escape(str(run_dir))}</p>
    <div class="metrics">{metrics_html}</div>
    <div class="panel">
      <img src="{escape(image_href)}" alt="Live training progress">
      <div class="links">
        <a href="{escape(image_href)}">Open image</a> |
        <a href="{escape(summary_href)}">Open summary</a>
      </div>
    </div>
  </div>
</body>
</html>"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def _open_live_dashboard(dashboard_path: Path) -> None:
    try:
        if hasattr(os, "startfile"):
            os.startfile(dashboard_path)  # type: ignore[attr-defined]
        else:
            webbrowser.open(dashboard_path.as_uri(), new=2)
    except OSError:
        pass


def resolve_amp_dtype(device: torch.device, amp_enabled: bool, amp_dtype: str) -> torch.dtype | None:
    if device.type != "cuda" or not amp_enabled:
        return None
    if amp_dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _optimizer_steps_per_epoch(num_train_batches: int, accumulation_steps: int) -> int:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    return max(1, math.ceil(num_train_batches / accumulation_steps))


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    clamped_warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0

    def lr_lambda(step: int) -> float:
        if clamped_warmup_steps > 0 and step < clamped_warmup_steps:
            return float(step + 1) / float(clamped_warmup_steps)

        if total_steps == clamped_warmup_steps:
            return max(min_lr_ratio, 1.0)

        decay_progress = (step - clamped_warmup_steps) / max(1, total_steps - clamped_warmup_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _sanitize_prediction_key(value: str) -> str:
    return (
        value.replace("^", "caret_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_task_specs(frame: pd.DataFrame, config: TransformerConfig) -> tuple[TaskSpec, ...]:
    target_columns = [config.primary_target_column, *config.auxiliary_target_columns]
    ordered_unique_columns = list(dict.fromkeys(target_columns))

    task_specs: list[TaskSpec] = []
    for target_column in ordered_unique_columns:
        if target_column not in frame.columns:
            raise ValueError(f"Dataset is missing required target column: {target_column}")

        labels = tuple(sorted(frame[target_column].dropna().unique().tolist()))
        if len(labels) < 2:
            raise ValueError(f"Target column {target_column} must contain at least two classes.")

        label_to_index, index_to_label = build_label_mapping(list(labels))
        task_specs.append(
            TaskSpec(
                name=target_column,
                target_column=target_column,
                labels=labels,
                label_to_index=label_to_index,
                index_to_label=index_to_label,
            )
        )
    return tuple(task_specs)


def _signal_strength(probabilities: np.ndarray, labels: tuple[str, ...]) -> float:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    if "up" not in label_to_index or "down" not in label_to_index:
        return 0.0
    return float(
        (probabilities[:, label_to_index["up"]] - probabilities[:, label_to_index["down"]]).mean()
    )


def _task_loss(
    outputs,
    targets: dict[str, torch.Tensor],
    criterion,
    primary_task: TaskSpec,
    auxiliary_tasks: tuple[TaskSpec, ...],
    auxiliary_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    primary_loss = criterion(outputs.logits_for(primary_task.name), targets[primary_task.name])
    if not auxiliary_tasks:
        zero = primary_loss.new_zeros(())
        return primary_loss, primary_loss, zero

    auxiliary_losses = torch.stack(
        [
            criterion(outputs.logits_for(task.name), targets[task.name])
            for task in auxiliary_tasks
        ]
    )
    auxiliary_loss = auxiliary_losses.mean()
    total_loss = primary_loss + auxiliary_loss_weight * auxiliary_loss
    return total_loss, primary_loss, auxiliary_loss


def _numpy_task_loss(
    outputs: TransformerPredictionOutput,
    criterion,
    primary_task: TaskSpec,
    auxiliary_tasks: tuple[TaskSpec, ...],
    auxiliary_loss_weight: float,
) -> tuple[float, float, float]:
    primary_loss = float(
        criterion(
            torch.from_numpy(outputs.task_logits[primary_task.name]),
            torch.from_numpy(outputs.task_targets[primary_task.name]),
        ).item()
    )
    if not auxiliary_tasks:
        return primary_loss, primary_loss, 0.0

    auxiliary_losses = [
        float(
            criterion(
                torch.from_numpy(outputs.task_logits[task.name]),
                torch.from_numpy(outputs.task_targets[task.name]),
            ).item()
        )
        for task in auxiliary_tasks
    ]
    auxiliary_loss = float(np.mean(auxiliary_losses))
    total_loss = primary_loss + auxiliary_loss_weight * auxiliary_loss
    return total_loss, primary_loss, auxiliary_loss


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    accumulation_steps: int,
    max_grad_norm: float,
    log_every_steps: int,
    primary_task: TaskSpec,
    auxiliary_tasks: tuple[TaskSpec, ...],
    auxiliary_loss_weight: float,
    global_step_offset: int = 0,
    run_start_time: float | None = None,
    live_update_callback=None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_auxiliary_loss = 0.0
    total_items = 0
    running_loss = 0.0
    running_primary_loss = 0.0
    running_auxiliary_loss = 0.0
    running_items = 0
    running_steps = 0
    use_cuda = device.type == "cuda"
    total_steps = len(loader)
    epoch_loop_start = time.perf_counter()
    last_log_time = epoch_loop_start

    optimizer.zero_grad(set_to_none=True)

    for step, (features, targets) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=use_cuda)
        targets = {
            task_name: target_tensor.to(device, non_blocking=use_cuda)
            for task_name, target_tensor in targets.items()
        }

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            outputs = model(features)
            loss, primary_loss, auxiliary_loss = _task_loss(
                outputs,
                targets,
                criterion,
                primary_task,
                auxiliary_tasks,
                auxiliary_loss_weight,
            )

        loss_to_backprop = loss / accumulation_steps
        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        should_step = step % accumulation_steps == 0 or step == total_steps
        if should_step:
            if max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = next(iter(targets.values())).size(0)
        batch_loss = loss.detach().item()
        batch_primary_loss = primary_loss.detach().item()
        batch_auxiliary_loss = auxiliary_loss.detach().item()
        total_loss += batch_loss * batch_size
        total_primary_loss += batch_primary_loss * batch_size
        total_auxiliary_loss += batch_auxiliary_loss * batch_size
        total_items += batch_size
        running_loss += batch_loss * batch_size
        running_primary_loss += batch_primary_loss * batch_size
        running_auxiliary_loss += batch_auxiliary_loss * batch_size
        running_items += batch_size
        running_steps += 1

        if log_every_steps > 0 and (step % log_every_steps == 0 or step == total_steps):
            current_time = time.perf_counter()
            interval_seconds = max(current_time - last_log_time, 1e-9)
            samples_per_second = running_items / interval_seconds
            batches_per_second = running_steps / interval_seconds
            global_step = global_step_offset + step
            step_summary = {
                "epoch": epoch,
                "step": step,
                "total_steps": total_steps,
                "global_step": global_step,
                "running_loss": running_loss / running_items,
                "running_primary_loss": running_primary_loss / running_items,
                "running_auxiliary_loss": running_auxiliary_loss / running_items,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "interval_seconds": interval_seconds,
                "samples_per_second": samples_per_second,
                "batches_per_second": batches_per_second,
                "elapsed_seconds": (
                    time.perf_counter() - run_start_time
                    if run_start_time is not None
                    else np.nan
                ),
            }
            telemetry = query_gpu_telemetry(
                device,
                elapsed_seconds=step_summary["elapsed_seconds"],
            )
            print(
                f"  Epoch {epoch:02d}/{total_epochs} step {step:03d}/{total_steps} | "
                f"loss={running_loss / running_items:.4f} | "
                f"primary={running_primary_loss / running_items:.4f} | "
                f"aux={running_auxiliary_loss / running_items:.4f}"
                f" | samples_s={samples_per_second:.1f}"
                + (
                    f" | gpu_util={telemetry['gpu_utilization_pct']:.0f}%"
                    f" | vram={telemetry['memory_used_gb']:.2f}GB"
                    + (
                        f" | gr_clk={telemetry['graphics_clock_mhz']:.0f}MHz"
                        if np.isfinite(telemetry["graphics_clock_mhz"])
                        else ""
                    )
                    + (
                        f" | power={telemetry['power_draw_w']:.1f}W"
                        if np.isfinite(telemetry["power_draw_w"])
                        else ""
                    )
                    if telemetry is not None
                    else ""
                ),
                flush=True,
            )
            if live_update_callback is not None:
                live_update_callback(step_summary, telemetry)
            running_loss = 0.0
            running_primary_loss = 0.0
            running_auxiliary_loss = 0.0
            running_items = 0
            running_steps = 0
            last_log_time = current_time

    epoch_seconds = max(time.perf_counter() - epoch_loop_start, 1e-9)
    return {
        "loss": total_loss / total_items,
        "primary_loss": total_primary_loss / total_items,
        "auxiliary_loss": total_auxiliary_loss / total_items,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "samples_per_second": total_items / epoch_seconds,
        "batches_per_second": total_steps / epoch_seconds,
    }


@torch.no_grad()
def _predict(
    model,
    loader,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None,
    task_specs: tuple[TaskSpec, ...],
) -> TransformerPredictionOutput:
    model.eval()
    task_logits_batches = {task.name: [] for task in task_specs}
    task_probability_batches = {task.name: [] for task in task_specs}
    regime_batches: list[np.ndarray] = []
    task_target_batches = {task.name: [] for task in task_specs}
    use_cuda = device.type == "cuda"
    primary_task_name = task_specs[0].name

    for features, targets in loader:
        features = features.to(device, non_blocking=use_cuda)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            outputs = model(features)

        for task in task_specs:
            task_logits_batches[task.name].append(outputs.logits_for(task.name).float().cpu().numpy())
            task_probability_batches[task.name].append(
                outputs.probability_distribution_for(task.name).float().cpu().numpy()
            )
            task_target_batches[task.name].append(targets[task.name].numpy())
        regime_batches.append(outputs.regime_state.float().cpu().numpy())

    task_logits = {
        task_name: np.concatenate(batches, axis=0)
        for task_name, batches in task_logits_batches.items()
    }
    task_probability_distributions = {
        task_name: np.concatenate(batches, axis=0)
        for task_name, batches in task_probability_batches.items()
    }
    task_targets = {
        task_name: np.concatenate(batches, axis=0)
        for task_name, batches in task_target_batches.items()
    }

    return TransformerPredictionOutput(
        logits=task_logits[primary_task_name],
        probability_distribution=task_probability_distributions[primary_task_name],
        regime_state=np.concatenate(regime_batches, axis=0),
        targets=task_targets[primary_task_name],
        task_logits=task_logits,
        task_probability_distributions=task_probability_distributions,
        task_targets=task_targets,
        primary_task_name=primary_task_name,
    )


def _evaluate_frame(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    predicted_column: str,
    labels: tuple[str, ...],
    probability_columns: list[str],
) -> dict[str, float]:
    metrics = evaluate_classifier_predictions(
        predictions[target_column],
        predictions[predicted_column],
        predictions.loc[:, probability_columns].to_numpy(),
        classes=labels,
    )
    metrics["average_signal_strength"] = _signal_strength(
        predictions.loc[:, probability_columns].to_numpy(),
        labels,
    )
    return metrics


def _slice_metrics(
    predictions: pd.DataFrame,
    event_column: str,
    *,
    target_column: str,
    predicted_column: str,
    labels: tuple[str, ...],
    probability_columns: list[str],
) -> dict[str, dict | int]:
    event_subset = predictions.loc[predictions[event_column] == 1]
    result: dict[str, dict | int] = {"row_count": int(len(event_subset))}
    if event_subset.empty:
        return result

    result["scores"] = _evaluate_frame(
        event_subset,
        target_column=target_column,
        predicted_column=predicted_column,
        labels=labels,
        probability_columns=probability_columns,
    )
    return result


def _ticker_slice_metrics(
    predictions: pd.DataFrame,
    target_ticker: str,
    *,
    target_column: str,
    predicted_column: str,
    labels: tuple[str, ...],
    probability_columns: list[str],
) -> dict[str, dict | int | str]:
    ticker_subset = predictions.loc[predictions["target_ticker"] == target_ticker]
    result: dict[str, dict | int | str] = {
        "target_ticker": target_ticker,
        "row_count": int(len(ticker_subset)),
    }
    if ticker_subset.empty:
        return result

    result["scores"] = _evaluate_frame(
        ticker_subset,
        target_column=target_column,
        predicted_column=predicted_column,
        labels=labels,
        probability_columns=probability_columns,
    )
    return result


def _deoverlapped_slice_metrics(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    predicted_column: str,
    labels: tuple[str, ...],
    probability_columns: list[str],
) -> dict[str, dict | int]:
    if "deoverlap_group_5d" not in predictions.columns:
        return {"row_count": 0}

    subset = predictions.loc[predictions["deoverlap_group_5d"] == 0]
    result: dict[str, dict | int] = {"row_count": int(len(subset))}
    if subset.empty:
        return result

    result["scores"] = _evaluate_frame(
        subset,
        target_column=target_column,
        predicted_column=predicted_column,
        labels=labels,
        probability_columns=probability_columns,
    )
    return result


def _prediction_frame(
    frame: pd.DataFrame,
    sequence_indices,
    outputs: TransformerPredictionOutput,
    primary_task: TaskSpec,
    auxiliary_tasks: tuple[TaskSpec, ...],
) -> pd.DataFrame:
    row_positions = [item.endpoint for item in sequence_indices]
    candidate_columns = [
        "date",
        "split",
        "target_ticker",
        primary_task.target_column,
        "target_ret_5d",
        "target_ret_1d",
        "target_excess_ret_5d",
        "target_excess_ret_1d",
        "vix_abs_10pct_flag",
        "vix_abs_20pct_flag",
        "deoverlap_group_5d",
    ]
    candidate_columns.extend(task.target_column for task in auxiliary_tasks)
    columns = [column for column in candidate_columns if column in frame.columns]

    base = frame.iloc[row_positions].loc[:, columns].copy()
    primary_probabilities = outputs.task_probability_distributions[primary_task.name]
    primary_predictions = primary_probabilities.argmax(axis=1)
    base["predicted_label"] = [
        primary_task.index_to_label[int(item)]
        for item in primary_predictions
    ]
    base["top_probability"] = primary_probabilities.max(axis=1)
    base["distribution_entropy"] = -np.sum(
        primary_probabilities * np.log(np.clip(primary_probabilities, 1e-12, 1.0)),
        axis=1,
    )
    base["regime_state_norm"] = np.linalg.norm(outputs.regime_state, axis=1)
    for class_index, class_label in primary_task.index_to_label.items():
        base[f"proba_{class_label}"] = primary_probabilities[:, class_index]

    for task in auxiliary_tasks:
        task_key = _sanitize_prediction_key(task.target_column)
        probabilities = outputs.task_probability_distributions[task.name]
        predictions = probabilities.argmax(axis=1)
        base[f"predicted_{task_key}"] = [
            task.index_to_label[int(item)]
            for item in predictions
        ]
        base[f"top_probability_{task_key}"] = probabilities.max(axis=1)
        base[f"distribution_entropy_{task_key}"] = -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
            axis=1,
        )
        for class_index, class_label in task.index_to_label.items():
            base[f"proba_{task_key}_{class_label}"] = probabilities[:, class_index]

    return base.reset_index(drop=True)


def train_transformer(
    frame: pd.DataFrame,
    run_dir: Path,
    config: TransformerConfig,
) -> dict:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    run_start_time = time.perf_counter()

    task_specs = _build_task_specs(frame, config)
    primary_task = task_specs[0]
    auxiliary_tasks = task_specs[1:]
    required_target_columns = [task.target_column for task in task_specs]
    frame = frame.dropna(subset=required_target_columns).reset_index(drop=True)

    feature_columns = get_feature_columns(frame)
    group_columns = ("target_ticker",) if "target_ticker" in frame.columns else None
    means, stds = fit_standardization_stats(frame.loc[frame["split"] == "train"], feature_columns)
    feature_matrix = standardize_feature_matrix(frame, feature_columns, means, stds)
    label_arrays = {
        task.name: frame[task.target_column].map(task.label_to_index).to_numpy(dtype=np.int64)
        for task in task_specs
    }
    metadata_columns = [
        "date",
        "split",
        "target_ticker",
        primary_task.target_column,
        *[task.target_column for task in auxiliary_tasks],
        "target_ret_5d",
        "target_ret_1d",
        "target_excess_ret_5d",
        "target_excess_ret_1d",
        "vix_abs_10pct_flag",
        "vix_abs_20pct_flag",
        "deoverlap_group_5d",
    ]
    metadata_frame = frame.loc[:, [column for column in metadata_columns if column in frame.columns]].copy()
    del frame

    train_indices = build_sequence_indices(
        metadata_frame,
        lookback=config.lookback,
        allowed_splits=("train",),
        group_columns=group_columns,
        window_stride=config.window_stride,
    )
    validation_indices = build_sequence_indices(
        metadata_frame,
        lookback=config.lookback,
        allowed_splits=("validation",),
        group_columns=group_columns,
        window_stride=config.window_stride,
    )
    test_indices = build_sequence_indices(
        metadata_frame,
        lookback=config.lookback,
        allowed_splits=("test",),
        group_columns=group_columns,
        window_stride=config.window_stride,
    )
    if config.loader_cache_mode not in {"auto", "ram", "disk"}:
        raise ValueError("loader_cache_mode must be one of: auto, ram, disk")

    requested_ram_loader = config.loader_cache_mode == "ram"
    requested_disk_loader = config.loader_cache_mode == "disk"
    auto_loader = config.loader_cache_mode == "auto"

    use_disk_loader_cache = False
    share_dataset_memory = False
    tensor_feature_matrix = None
    tensor_label_arrays = None
    feature_matrix_path = None
    label_array_paths = None

    should_try_ram_loader = config.num_workers > 0 and (requested_ram_loader or auto_loader)
    if should_try_ram_loader:
        try:
            tensor_feature_matrix, tensor_label_arrays = _build_tensor_backed_arrays(
                feature_matrix,
                label_arrays,
                share_memory=config.num_workers > 0,
            )
            share_dataset_memory = True
            print(
                f"Prepared RAM-backed loader tensors for {config.num_workers} worker processes.",
                flush=True,
            )
        except RuntimeError as exc:
            if requested_ram_loader:
                raise RuntimeError(
                    "RAM-backed multi-worker loading failed. Reduce workers or use --loader-cache-mode disk."
                ) from exc
            use_disk_loader_cache = True

    if requested_disk_loader:
        use_disk_loader_cache = True
        tensor_feature_matrix = None
        tensor_label_arrays = None
        share_dataset_memory = False

    if use_disk_loader_cache:
        loader_cache_dir = run_dir / "loader_cache"
        loader_cache_dir.mkdir(parents=True, exist_ok=True)
        feature_matrix_path = loader_cache_dir / "feature_matrix.npy"
        np.save(feature_matrix_path, feature_matrix, allow_pickle=False)
        label_array_paths = {}
        for task_name, values in label_arrays.items():
            target_path = loader_cache_dir / f"{_sanitize_prediction_key(task_name)}_labels.npy"
            np.save(target_path, values, allow_pickle=False)
            label_array_paths[task_name] = target_path
        print(
            f"Prepared disk-backed loader cache at {loader_cache_dir} for multi-worker Windows loading.",
            flush=True,
        )

    train_dataset = MultiTaskRollingWindowDataset(
        frame=None,
        feature_columns=feature_columns,
        task_target_columns=None,
        task_label_to_index=None,
        sequence_indices=train_indices,
        lookback=config.lookback,
        feature_matrix=tensor_feature_matrix if tensor_feature_matrix is not None else (feature_matrix if feature_matrix_path is None else None),
        label_arrays=tensor_label_arrays if tensor_label_arrays is not None else (label_arrays if label_array_paths is None else None),
        feature_matrix_path=feature_matrix_path,
        label_array_paths=label_array_paths,
        share_memory=False,
    )
    validation_dataset = MultiTaskRollingWindowDataset(
        frame=None,
        feature_columns=feature_columns,
        task_target_columns=None,
        task_label_to_index=None,
        sequence_indices=validation_indices,
        lookback=config.lookback,
        feature_matrix=tensor_feature_matrix if tensor_feature_matrix is not None else (feature_matrix if feature_matrix_path is None else None),
        label_arrays=tensor_label_arrays if tensor_label_arrays is not None else (label_arrays if label_array_paths is None else None),
        feature_matrix_path=feature_matrix_path,
        label_array_paths=label_array_paths,
        share_memory=False,
    )
    test_dataset = MultiTaskRollingWindowDataset(
        frame=None,
        feature_columns=feature_columns,
        task_target_columns=None,
        task_label_to_index=None,
        sequence_indices=test_indices,
        lookback=config.lookback,
        feature_matrix=tensor_feature_matrix if tensor_feature_matrix is not None else (feature_matrix if feature_matrix_path is None else None),
        label_arrays=tensor_label_arrays if tensor_label_arrays is not None else (label_arrays if label_array_paths is None else None),
        feature_matrix_path=feature_matrix_path,
        label_array_paths=label_array_paths,
        share_memory=False,
    )

    device = choose_device(config.device)
    configure_runtime(device)
    amp_dtype = resolve_amp_dtype(device, config.amp, config.amp_dtype)
    pin_memory = device.type == "cuda"

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = PatchTransformerClassifier(
        feature_dim=len(feature_columns),
        sequence_length=config.lookback,
        task_output_dims={task.name: len(task.labels) for task in task_specs},
        primary_task_name=primary_task.name,
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        regime_dim=config.regime_dim,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        ff_dim=config.ff_dim,
        norm_first=config.norm_first,
        use_revin=config.use_revin,
        revin_affine=config.revin_affine,
    ).to(device)
    total_parameters, trainable_parameters = count_model_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), config.accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_fraction)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )

    gpu_name = ""
    gpu_vram = ""
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = f" | gpu={properties.name}"
        gpu_vram = f" | vram_gb={properties.total_memory / (1024 ** 3):.1f}"
    amp_description = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "disabled"
    auxiliary_label = ", ".join(task.target_column for task in auxiliary_tasks) or "none"

    print(
        f"Device: {device} | train_windows={len(train_dataset)} "
        f"validation_windows={len(validation_dataset)} test_windows={len(test_dataset)} | "
        f"feature_dim={len(feature_columns)}{gpu_name}{gpu_vram} | amp={amp_description} | "
        f"primary_target={primary_task.target_column} | aux_targets={auxiliary_label} | "
        f"params={total_parameters:,} | regime_dim={config.regime_dim} | ff_dim={config.ff_dim or config.d_model * 4} | "
        f"window_stride={config.window_stride} | batch_size={config.batch_size} | "
        f"lr={config.learning_rate:.2e} | warmup_steps={warmup_steps} | min_lr_ratio={config.min_lr_ratio:.2f} | "
        f"effective_batch={config.batch_size * config.accumulation_steps} | "
        f"num_workers={config.num_workers}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    step_history: list[dict[str, float]] = []
    telemetry_history: list[dict[str, float]] = []
    best_checkpoint_path = run_dir / "best_model.pt"
    live_plot_path = run_dir / "training_progress.png"
    live_history_path = run_dir / "training_history.json"
    live_status_path = run_dir / "live_status.json"
    dashboard_path = run_dir / "dashboard.html"
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    global_step_offset = 0

    def write_live_artifacts(status: dict) -> None:
        plot_training_curves(
            history,
            live_plot_path,
            step_history=step_history,
            telemetry_history=telemetry_history,
        )
        write_json(
            {
                "config": asdict(config),
                "best_epoch_so_far": status.get("best_epoch_so_far"),
                "history": history,
                "step_history": step_history,
                "step_summary": summarize_step_history(step_history),
                "telemetry_history": telemetry_history,
                "telemetry_summary": summarize_telemetry_history(telemetry_history),
                "latest_epoch": status.get("latest_epoch"),
                "latest_step": status.get("latest_step"),
                "latest_samples_per_second": status.get("latest_samples_per_second"),
                "phase": status.get("phase"),
            },
            live_history_path,
        )
        write_json(status, live_status_path)
        _write_live_dashboard(
            dashboard_path=dashboard_path,
            live_plot_path=live_plot_path,
            run_dir=run_dir,
            status=status,
        )

    print(
        f"Live progress files: plot={live_plot_path} | history={live_history_path} | dashboard={dashboard_path}",
        flush=True,
    )
    initial_telemetry = query_gpu_telemetry(device, elapsed_seconds=0.0)
    if initial_telemetry is not None:
        telemetry_history.append(initial_telemetry)
    initial_status = {
        "phase": "starting",
        "latest_epoch": None,
        "total_epochs": config.epochs,
        "latest_step": None,
        "total_steps": len(train_loader),
        "best_epoch_so_far": None,
        "latest_running_loss": None,
        "latest_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "latest_samples_per_second": None,
        "latest_gpu": initial_telemetry,
    }
    write_live_artifacts(initial_status)
    if config.open_dashboard:
        _open_live_dashboard(dashboard_path)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        def on_live_step(step_summary: dict[str, float], telemetry: dict[str, float] | None) -> None:
            step_history.append(step_summary)
            if telemetry is not None:
                telemetry_history.append(telemetry)
            write_live_artifacts(
                {
                    "phase": "training",
                    "latest_epoch": epoch,
                    "total_epochs": config.epochs,
                    "latest_step": int(step_summary["step"]),
                    "total_steps": int(step_summary["total_steps"]),
                    "best_epoch_so_far": best_epoch if best_epoch > 0 else None,
                    "latest_running_loss": float(step_summary["running_loss"]),
                    "latest_learning_rate": float(step_summary["learning_rate"]),
                    "latest_samples_per_second": float(step_summary["samples_per_second"]),
                    "latest_gpu": telemetry,
                }
            )

        train_losses = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch=epoch,
            total_epochs=config.epochs,
            amp_dtype=amp_dtype,
            scaler=scaler,
            accumulation_steps=config.accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            log_every_steps=config.log_every_steps,
            primary_task=primary_task,
            auxiliary_tasks=auxiliary_tasks,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
            global_step_offset=global_step_offset,
            run_start_time=run_start_time,
            live_update_callback=on_live_step,
        )
        global_step_offset += len(train_loader)
        val_outputs = _predict(model, validation_loader, device, amp_dtype=amp_dtype, task_specs=task_specs)
        val_loss, val_primary_loss, val_auxiliary_loss = _numpy_task_loss(
            val_outputs,
            criterion,
            primary_task,
            auxiliary_tasks,
            config.auxiliary_loss_weight,
        )
        val_probabilities = val_outputs.probability_distribution
        val_predictions = val_probabilities.argmax(axis=1)
        val_labels = np.array([primary_task.index_to_label[int(item)] for item in val_outputs.targets])
        val_predicted_labels = np.array(
            [primary_task.index_to_label[int(item)] for item in val_predictions]
        )
        val_predicted_shares = {
            label: float(np.mean(val_predicted_labels == label))
            for label in primary_task.labels
        }
        val_metrics = evaluate_classifier_predictions(
            val_labels,
            val_predicted_labels,
            val_probabilities,
            classes=primary_task.labels,
        )

        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "train_primary_loss": train_losses["primary_loss"],
                "train_auxiliary_loss": train_losses["auxiliary_loss"],
                "learning_rate": train_losses["learning_rate"],
                "train_samples_per_second": train_losses["samples_per_second"],
                "train_batches_per_second": train_losses["batches_per_second"],
                "val_loss": val_loss,
                "val_primary_loss": val_primary_loss,
                "val_auxiliary_loss": val_auxiliary_loss,
                "val_log_loss": val_metrics["log_loss"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_directional_hit_rate": val_metrics["directional_hit_rate"],
                "epoch_seconds": epoch_seconds,
                **{f"val_pred_share_{label}": share for label, share in val_predicted_shares.items()},
            }
        )
        best_epoch_so_far = epoch if val_loss < best_val_loss else best_epoch
        epoch_telemetry = query_gpu_telemetry(
            device,
            elapsed_seconds=time.perf_counter() - run_start_time,
        )
        if epoch_telemetry is not None:
            telemetry_history.append(epoch_telemetry)
        write_live_artifacts(
            {
                "phase": "validating",
                "latest_epoch": epoch,
                "total_epochs": config.epochs,
                "latest_step": len(train_loader),
                "total_steps": len(train_loader),
                "best_epoch_so_far": best_epoch_so_far if best_epoch_so_far > 0 else None,
                "latest_running_loss": float(train_losses["loss"]),
                "latest_learning_rate": float(train_losses["learning_rate"]),
                "latest_samples_per_second": float(train_losses["samples_per_second"]),
                "latest_gpu": epoch_telemetry,
            }
        )
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_losses['loss']:.4f} | "
            f"train_primary={train_losses['primary_loss']:.4f} | "
            f"train_aux={train_losses['auxiliary_loss']:.4f} | "
            f"lr={train_losses['learning_rate']:.2e} | "
            f"train_samples_s={train_losses['samples_per_second']:.1f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_primary={val_primary_loss:.4f} | "
            f"val_aux={val_auxiliary_loss:.4f} | "
            f"val_log_loss={val_metrics['log_loss']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_dir_hit={val_metrics['directional_hit_rate']:.4f} | "
            f"pred_mix="
            f"{', '.join(f'{label}:{val_predicted_shares[label]:.2f}' for label in primary_task.labels)} | "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(best_state, best_checkpoint_path)
            print(f"  New best model saved at epoch {epoch}.", flush=True)
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(
                    f"  Early stopping triggered after epoch {epoch} "
                    f"(best epoch: {best_epoch}).",
                    flush=True,
                )
                break

    best_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state["model"])

    val_outputs = _predict(model, validation_loader, device, amp_dtype=amp_dtype, task_specs=task_specs)
    test_outputs = _predict(model, test_loader, device, amp_dtype=amp_dtype, task_specs=task_specs)

    def build_split_result(
        sequence_indices,
        outputs: TransformerPredictionOutput,
        split_name: str,
    ) -> tuple[dict, pd.DataFrame]:
        pred_frame = _prediction_frame(
            metadata_frame,
            sequence_indices,
            outputs,
            primary_task,
            auxiliary_tasks,
        )

        probability_columns = [f"proba_{label}" for label in primary_task.labels]
        metrics = _evaluate_frame(
            pred_frame,
            target_column=primary_task.target_column,
            predicted_column="predicted_label",
            labels=primary_task.labels,
            probability_columns=probability_columns,
        )
        metrics["average_top_probability"] = float(pred_frame["top_probability"].mean())
        metrics["average_distribution_entropy"] = float(pred_frame["distribution_entropy"].mean())
        metrics["average_regime_state_norm"] = float(pred_frame["regime_state_norm"].mean())
        metrics["event_slices"] = {
            "vix_abs_10pct_flag": _slice_metrics(
                pred_frame,
                "vix_abs_10pct_flag",
                target_column=primary_task.target_column,
                predicted_column="predicted_label",
                labels=primary_task.labels,
                probability_columns=probability_columns,
            ),
            "vix_abs_20pct_flag": _slice_metrics(
                pred_frame,
                "vix_abs_20pct_flag",
                target_column=primary_task.target_column,
                predicted_column="predicted_label",
                labels=primary_task.labels,
                probability_columns=probability_columns,
            ),
        }
        metrics["deoverlapped_slice"] = _deoverlapped_slice_metrics(
            pred_frame,
            target_column=primary_task.target_column,
            predicted_column="predicted_label",
            labels=primary_task.labels,
            probability_columns=probability_columns,
        )
        if "target_ticker" in pred_frame.columns:
            metrics["ticker_slices"] = {
                config.focus_ticker: _ticker_slice_metrics(
                    pred_frame,
                    config.focus_ticker,
                    target_column=primary_task.target_column,
                    predicted_column="predicted_label",
                    labels=primary_task.labels,
                    probability_columns=probability_columns,
                ),
            }

        if auxiliary_tasks:
            metrics["auxiliary_tasks"] = {}
            for task in auxiliary_tasks:
                task_key = _sanitize_prediction_key(task.target_column)
                task_probability_columns = [
                    f"proba_{task_key}_{label}"
                    for label in task.labels
                ]
                auxiliary_scores = _evaluate_frame(
                    pred_frame,
                    target_column=task.target_column,
                    predicted_column=f"predicted_{task_key}",
                    labels=task.labels,
                    probability_columns=task_probability_columns,
                )
                auxiliary_scores["average_top_probability"] = float(
                    pred_frame[f"top_probability_{task_key}"].mean()
                )
                auxiliary_scores["average_distribution_entropy"] = float(
                    pred_frame[f"distribution_entropy_{task_key}"].mean()
                )
                metrics["auxiliary_tasks"][task.target_column] = auxiliary_scores

        pred_frame.to_parquet(run_dir / f"{split_name}_predictions.parquet", index=False)
        return metrics, pred_frame

    validation_metrics, validation_predictions = build_split_result(
        validation_indices,
        val_outputs,
        "validation",
    )
    test_metrics, test_predictions = build_split_result(test_indices, test_outputs, "test")

    final_telemetry = query_gpu_telemetry(
        device,
        elapsed_seconds=time.perf_counter() - run_start_time,
    )
    if final_telemetry is not None:
        telemetry_history.append(final_telemetry)

    plot_training_curves(
        history,
        run_dir / "training_curves.png",
        step_history=step_history,
        telemetry_history=telemetry_history,
    )
    plot_confusion_matrix(
        y_true=test_predictions[primary_task.target_column].to_numpy(),
        y_pred=test_predictions["predicted_label"].to_numpy(),
        labels=list(primary_task.labels),
        output_path=run_dir / "test_confusion_matrix.png",
    )

    normalizer_payload = {
        "feature_columns": feature_columns,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "primary_target_column": primary_task.target_column,
        "auxiliary_target_columns": [task.target_column for task in auxiliary_tasks],
    }
    joblib.dump(normalizer_payload, run_dir / "normalizer.joblib")

    summary = {
        "config": asdict(config),
        "device": str(device),
        "best_epoch": best_epoch,
        "model_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "task_specs": {
            task.target_column: {"labels": list(task.labels)}
            for task in task_specs
        },
        "history": history,
        "step_summary": summarize_step_history(step_history),
        "telemetry_summary": summarize_telemetry_history(telemetry_history),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(summary, run_dir / "summary.json")
    write_live_artifacts(
        {
            "phase": "completed",
            "latest_epoch": history[-1]["epoch"] if history else None,
            "total_epochs": config.epochs,
            "latest_step": len(train_loader),
            "total_steps": len(train_loader),
            "best_epoch_so_far": best_epoch if best_epoch > 0 else None,
            "latest_running_loss": float(history[-1]["train_loss"]) if history else None,
            "latest_learning_rate": float(history[-1]["learning_rate"]) if history else None,
            "latest_samples_per_second": float(history[-1]["train_samples_per_second"]) if history else None,
            "latest_gpu": final_telemetry,
        }
    )
    del train_loader, validation_loader, test_loader
    del train_dataset, validation_dataset, test_dataset
    del train_indices, validation_indices, test_indices
    del feature_matrix, label_arrays, metadata_frame
    del model, optimizer, scheduler, scaler
    del val_outputs, test_outputs
    del validation_predictions, test_predictions
    cleanup_runtime(device)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the compact patch-based transformer model.")
    parser.add_argument(
        "--dataset-path",
        default=str(PROCESSED_DATASET_PATH),
        help="Processed dataset parquet path.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional explicit run directory. If omitted, a timestamped run directory is created.",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="Step size between consecutive window endpoints. Use >1 to reduce overlap on daily data.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Fraction of optimizer steps used for linear warmup.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Final cosine-decayed learning rate as a fraction of the base learning rate.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--regime-dim", type=int, default=32)
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=None,
        help="Transformer feed-forward hidden width. Defaults to 4x d-model when omitted.",
    )
    parser.add_argument("--patch-length", type=int, default=5)
    parser.add_argument("--patch-stride", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Attention-weight dropout inside the transformer blocks.",
    )
    parser.add_argument(
        "--post-norm",
        action="store_true",
        help="Use post-norm transformer blocks instead of the default pre-norm blocks.",
    )
    parser.add_argument(
        "--use-revin",
        action="store_true",
        help="Apply RevIN normalization to each input window before patching.",
    )
    parser.add_argument(
        "--no-revin-affine",
        action="store_true",
        help="Disable RevIN affine scale/bias parameters when RevIN is enabled.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device selection.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count. Defaults to 2 for larger Windows GPU runs.",
    )
    parser.add_argument(
        "--loader-cache-mode",
        choices=("auto", "ram", "disk"),
        default="auto",
        help="Loader backend for multi-worker CPU data. 'ram' avoids disk caches, 'disk' forces memmaps.",
    )
    parser.add_argument(
        "--no-open-dashboard",
        action="store_true",
        help="Do not auto-open the live HTML dashboard when training starts.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bfloat16", "float16"),
        default="auto",
        help="AMP dtype when CUDA mixed precision is enabled.",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for larger effective batch sizes.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold. Set to 0 to disable.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=10,
        help="Batch-step progress logging interval within each epoch.",
    )
    parser.add_argument(
        "--primary-target-column",
        default="target_label_5d",
        help="Primary classification target column.",
    )
    parser.add_argument(
        "--aux-target-columns",
        default="",
        help="Comma-separated auxiliary classification target columns.",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.5,
        help="Weight applied to the mean auxiliary loss.",
    )
    parser.add_argument(
        "--focus-ticker",
        default="SPY",
        help="Ticker to highlight in grouped-panel evaluation slices.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = TransformerConfig(
        lookback=args.lookback,
        window_stride=args.window_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        min_lr_ratio=args.min_lr_ratio,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        regime_dim=args.regime_dim,
        ff_dim=args.ff_dim,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        norm_first=not args.post_norm,
        use_revin=args.use_revin,
        revin_affine=not args.no_revin_affine,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        loader_cache_mode=args.loader_cache_mode,
        amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_every_steps=args.log_every_steps,
        primary_target_column=args.primary_target_column,
        auxiliary_target_columns=_parse_csv(args.aux_target_columns),
        auxiliary_loss_weight=args.aux_loss_weight,
        focus_ticker=args.focus_ticker,
        open_dashboard=not args.no_open_dashboard,
    )

    frame = read_parquet(args.dataset_path)
    prefix = "transformer-multitask" if config.auxiliary_target_columns else "transformer"
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = create_run_dir(prefix=prefix)
    summary = train_transformer(frame=frame, run_dir=run_dir, config=config)

    print(f"Saved transformer run to {run_dir}", flush=True)
    print(
        "Final validation/test: "
        f"val_log_loss={summary['validation']['log_loss']:.4f}, "
        f"val_bal_acc={summary['validation']['balanced_accuracy']:.4f}, "
        f"test_log_loss={summary['test']['log_loss']:.4f}, "
        f"test_bal_acc={summary['test']['balanced_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
