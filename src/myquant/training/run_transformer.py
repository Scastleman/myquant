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
    RollingWindowDataset,
    apply_standardization,
    build_label_mapping,
    build_sequence_indices,
    fit_standardization_stats,
    get_feature_columns,
)


@dataclass(frozen=True)
class TransformerConfig:
    lookback: int = 60
    patch_length: int = 5
    patch_stride: int = 5
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
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
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    running_loss = 0.0
    running_items = 0
    use_cuda = device.type == "cuda"
    total_steps = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for step, (features, targets) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            logits = model(features)
            loss = criterion(logits, targets)

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

        batch_size = targets.size(0)
        batch_loss = loss.detach().item()
        total_loss += batch_loss * batch_size
        total_items += batch_size
        running_loss += batch_loss * batch_size
        running_items += batch_size

        if log_every_steps > 0 and (step % log_every_steps == 0 or step == total_steps):
            print(
                f"  Epoch {epoch:02d}/{total_epochs} step {step:03d}/{total_steps} | "
                f"loss={running_loss / running_items:.4f}",
                flush=True,
            )
            running_loss = 0.0
            running_items = 0

    return total_loss / total_items


@torch.no_grad()
def _predict(
    model,
    loader,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    use_cuda = device.type == "cuda"

    for features, targets in loader:
        features = features.to(device, non_blocking=use_cuda)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            logits = model(features)
        logits_batches.append(logits.float().cpu().numpy())
        target_batches.append(targets.numpy())

    logits = np.concatenate(logits_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return logits, targets


def _probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(logits)
    probabilities = torch.softmax(tensor, dim=1).numpy()
    return probabilities


def _prediction_frame(
    frame: pd.DataFrame,
    sequence_indices,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
) -> pd.DataFrame:
    row_positions = [item.endpoint for item in sequence_indices]
    columns = [
        "date",
        "split",
        "target_label_5d",
        "target_ret_5d",
        "target_ret_1d",
        "vix_abs_10pct_flag",
        "vix_abs_20pct_flag",
    ]
    if "target_ticker" in frame.columns:
        columns.insert(2, "target_ticker")

    base = frame.iloc[row_positions].loc[:, columns].copy()
    base["predicted_label"] = [index_to_label[int(item)] for item in predicted_labels]
    for class_index, class_label in index_to_label.items():
        base[f"proba_{class_label}"] = probabilities[:, class_index]
    return base.reset_index(drop=True)


def _slice_metrics(predictions: pd.DataFrame, event_column: str, labels: list[str]) -> dict[str, dict | int]:
    event_subset = predictions.loc[predictions[event_column] == 1]
    result: dict[str, dict | int] = {"row_count": int(len(event_subset))}
    if event_subset.empty:
        return result

    probability_columns = [f"proba_{label}" for label in labels]
    scores = evaluate_classifier_predictions(
        event_subset["target_label_5d"],
        event_subset["predicted_label"],
        event_subset.loc[:, probability_columns].to_numpy(),
        classes=labels,
    )
    scores["average_signal_strength"] = float(
        (event_subset["proba_up"] - event_subset["proba_down"]).mean()
    )
    result["scores"] = scores
    return result


def _target_ticker_slice_metrics(
    predictions: pd.DataFrame,
    target_ticker: str,
    labels: list[str],
) -> dict[str, dict | int | str]:
    ticker_subset = predictions.loc[predictions["target_ticker"] == target_ticker]
    result: dict[str, dict | int | str] = {
        "target_ticker": target_ticker,
        "row_count": int(len(ticker_subset)),
    }
    if ticker_subset.empty:
        return result

    probability_columns = [f"proba_{label}" for label in labels]
    scores = evaluate_classifier_predictions(
        ticker_subset["target_label_5d"],
        ticker_subset["predicted_label"],
        ticker_subset.loc[:, probability_columns].to_numpy(),
        classes=labels,
    )
    scores["average_signal_strength"] = float(
        (ticker_subset["proba_up"] - ticker_subset["proba_down"]).mean()
    )
    result["scores"] = scores
    return result


def train_transformer(
    frame: pd.DataFrame,
    run_dir: Path,
    config: TransformerConfig,
) -> dict:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    labels = sorted(frame["target_label_5d"].unique().tolist())
    label_to_index, index_to_label = build_label_mapping(labels)
    feature_columns = get_feature_columns(frame)
    group_columns = ("target_ticker",) if "target_ticker" in frame.columns else None
    means, stds = fit_standardization_stats(frame.loc[frame["split"] == "train"], feature_columns)
    normalized = apply_standardization(frame, feature_columns, means, stds)

    train_indices = build_sequence_indices(
        normalized,
        lookback=config.lookback,
        allowed_splits=("train",),
        group_columns=group_columns,
    )
    validation_indices = build_sequence_indices(
        normalized,
        lookback=config.lookback,
        allowed_splits=("validation",),
        group_columns=group_columns,
    )
    test_indices = build_sequence_indices(
        normalized,
        lookback=config.lookback,
        allowed_splits=("test",),
        group_columns=group_columns,
    )

    train_dataset = RollingWindowDataset(normalized, feature_columns, label_to_index, train_indices, config.lookback)
    validation_dataset = RollingWindowDataset(
        normalized,
        feature_columns,
        label_to_index,
        validation_indices,
        config.lookback,
    )
    test_dataset = RollingWindowDataset(normalized, feature_columns, label_to_index, test_indices, config.lookback)

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
        num_classes=len(labels),
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
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

    print(
        f"Device: {device} | train_windows={len(train_dataset)} "
        f"validation_windows={len(validation_dataset)} test_windows={len(test_dataset)} | "
        f"feature_dim={len(feature_columns)}{gpu_name}{gpu_vram} | amp={amp_description} | "
        f"batch_size={config.batch_size} | effective_batch={config.batch_size * config.accumulation_steps} | "
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
        train_loss = _run_epoch(
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
        )
        val_logits, val_targets = _predict(model, validation_loader, device, amp_dtype=amp_dtype)
        val_loss = float(criterion(torch.from_numpy(val_logits), torch.from_numpy(val_targets)).item())
        val_probabilities = _probabilities_from_logits(val_logits)
        val_predictions = val_probabilities.argmax(axis=1)
        val_labels = np.array([index_to_label[int(item)] for item in val_targets])
        val_predicted_labels = np.array([index_to_label[int(item)] for item in val_predictions])
        val_metrics = evaluate_classifier_predictions(
            val_labels,
            val_predicted_labels,
            val_probabilities,
            classes=labels,
        )

        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_log_loss": val_metrics["log_loss"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_directional_hit_rate": val_metrics["directional_hit_rate"],
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
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

    val_logits, val_targets = _predict(model, validation_loader, device, amp_dtype=amp_dtype)
    test_logits, test_targets = _predict(model, test_loader, device, amp_dtype=amp_dtype)

    def build_split_result(sequence_indices, logits, targets, split_name: str) -> tuple[dict, pd.DataFrame]:
        probabilities = _probabilities_from_logits(logits)
        predictions = probabilities.argmax(axis=1)
        true_labels = np.array([index_to_label[int(item)] for item in targets])
        predicted_labels = np.array([index_to_label[int(item)] for item in predictions])
        metrics = evaluate_classifier_predictions(true_labels, predicted_labels, probabilities, classes=labels)
        pred_frame = _prediction_frame(frame, sequence_indices, predictions, probabilities, index_to_label)
        metrics["average_signal_strength"] = float((pred_frame["proba_up"] - pred_frame["proba_down"]).mean())
        metrics["event_slices"] = {
            "vix_abs_10pct_flag": _slice_metrics(pred_frame, "vix_abs_10pct_flag", labels),
            "vix_abs_20pct_flag": _slice_metrics(pred_frame, "vix_abs_20pct_flag", labels),
        }
        if "target_ticker" in pred_frame.columns:
            metrics["ticker_slices"] = {
                "SPY": _target_ticker_slice_metrics(pred_frame, "SPY", labels),
            }
        pred_frame.to_parquet(run_dir / f"{split_name}_predictions.parquet", index=False)
        return metrics, pred_frame

    validation_metrics, validation_predictions = build_split_result(
        validation_indices,
        val_logits,
        val_targets,
        "validation",
    )
    test_metrics, test_predictions = build_split_result(test_indices, test_logits, test_targets, "test")

    plot_training_curves(history, run_dir / "training_curves.png")
    plot_confusion_matrix(
        y_true=test_predictions["target_label_5d"].to_numpy(),
        y_pred=test_predictions["predicted_label"].to_numpy(),
        labels=labels,
        output_path=run_dir / "test_confusion_matrix.png",
    )

    normalizer_payload = {
        "feature_columns": feature_columns,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
    }
    joblib.dump(normalizer_payload, run_dir / "normalizer.joblib")

    summary = {
        "config": asdict(config),
        "device": str(device),
        "best_epoch": best_epoch,
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
    )

    frame = read_parquet(args.dataset_path)
    run_dir = create_run_dir(prefix="transformer")
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
