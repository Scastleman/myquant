from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    patch_length: int = 5
    patch_stride: int = 5
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    regime_dim: int = 32
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    device: str = "auto"
    num_workers: int = 2
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
    use_cuda = device.type == "cuda"
    total_steps = len(loader)

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

        if log_every_steps > 0 and (step % log_every_steps == 0 or step == total_steps):
            print(
                f"  Epoch {epoch:02d}/{total_epochs} step {step:03d}/{total_steps} | "
                f"loss={running_loss / running_items:.4f} | "
                f"primary={running_primary_loss / running_items:.4f} | "
                f"aux={running_auxiliary_loss / running_items:.4f}",
                flush=True,
            )
            running_loss = 0.0
            running_primary_loss = 0.0
            running_auxiliary_loss = 0.0
            running_items = 0

    return {
        "loss": total_loss / total_items,
        "primary_loss": total_primary_loss / total_items,
        "auxiliary_loss": total_auxiliary_loss / total_items,
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
    )
    validation_indices = build_sequence_indices(
        metadata_frame,
        lookback=config.lookback,
        allowed_splits=("validation",),
        group_columns=group_columns,
    )
    test_indices = build_sequence_indices(
        metadata_frame,
        lookback=config.lookback,
        allowed_splits=("test",),
        group_columns=group_columns,
    )

    train_dataset = MultiTaskRollingWindowDataset(
        frame=None,
        feature_columns=feature_columns,
        task_target_columns=None,
        task_label_to_index=None,
        sequence_indices=train_indices,
        lookback=config.lookback,
        feature_matrix=feature_matrix,
        label_arrays=label_arrays,
    )
    validation_dataset = MultiTaskRollingWindowDataset(
        frame=None,
        feature_columns=feature_columns,
        task_target_columns=None,
        task_label_to_index=None,
        sequence_indices=validation_indices,
        lookback=config.lookback,
        feature_matrix=feature_matrix,
        label_arrays=label_arrays,
    )
    test_dataset = MultiTaskRollingWindowDataset(
        frame=None,
        feature_columns=feature_columns,
        task_target_columns=None,
        task_label_to_index=None,
        sequence_indices=test_indices,
        lookback=config.lookback,
        feature_matrix=feature_matrix,
        label_arrays=label_arrays,
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
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
        f"regime_dim={config.regime_dim} | batch_size={config.batch_size} | "
        f"effective_batch={config.batch_size * config.accumulation_steps} | "
        f"num_workers={config.num_workers}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    best_checkpoint_path = run_dir / "best_model.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_losses = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
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
        )
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
                "val_loss": val_loss,
                "val_primary_loss": val_primary_loss,
                "val_auxiliary_loss": val_auxiliary_loss,
                "val_log_loss": val_metrics["log_loss"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_directional_hit_rate": val_metrics["directional_hit_rate"],
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_losses['loss']:.4f} | "
            f"train_primary={train_losses['primary_loss']:.4f} | "
            f"train_aux={train_losses['auxiliary_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_primary={val_primary_loss:.4f} | "
            f"val_aux={val_auxiliary_loss:.4f} | "
            f"val_log_loss={val_metrics['log_loss']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_dir_hit={val_metrics['directional_hit_rate']:.4f} | "
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

    plot_training_curves(history, run_dir / "training_curves.png")
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
        "task_specs": {
            task.target_column: {"labels": list(task.labels)}
            for task in task_specs
        },
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the compact patch-based transformer model.")
    parser.add_argument(
        "--dataset-path",
        default=str(PROCESSED_DATASET_PATH),
        help="Processed dataset parquet path.",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--regime-dim", type=int, default=32)
    parser.add_argument("--patch-length", type=int, default=5)
    parser.add_argument("--patch-stride", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
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
        help="DataLoader worker count. Use 2-4 on Windows for larger GPU runs.",
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        regime_dim=args.regime_dim,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_every_steps=args.log_every_steps,
        primary_target_column=args.primary_target_column,
        auxiliary_target_columns=_parse_csv(args.aux_target_columns),
        auxiliary_loss_weight=args.aux_loss_weight,
        focus_ticker=args.focus_ticker,
    )

    frame = read_parquet(args.dataset_path)
    prefix = "transformer-multitask" if config.auxiliary_target_columns else "transformer"
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
