from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_curves(
    history: list[dict[str, float]],
    output_path: str | Path,
    *,
    step_history: list[dict[str, float]] | None = None,
    telemetry_history: list[dict[str, float]] | None = None,
) -> Path:
    """Save a compact training dashboard that updates cleanly during long runs."""
    step_history = step_history or []
    telemetry_history = telemetry_history or []

    has_epoch_history = bool(history)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_primary_loss = [item.get("train_primary_loss", np.nan) for item in history]
    val_primary_loss = [item.get("val_primary_loss", np.nan) for item in history]
    val_bal_acc = [item["val_balanced_accuracy"] for item in history]
    val_dir_hit = [item.get("val_directional_hit_rate", np.nan) for item in history]
    learning_rate = [item.get("learning_rate", np.nan) for item in history]

    predicted_share_keys = sorted(
        {
            key.removeprefix("val_pred_share_")
            for item in history
            for key in item
            if key.startswith("val_pred_share_")
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    ax_loss = axes[0, 0]
    ax_metric = axes[0, 1]
    ax_lr = axes[1, 0]
    ax_share = axes[1, 1]
    ax_step = axes[2, 0]
    ax_gpu = axes[2, 1]

    if has_epoch_history:
        ax_loss.plot(epochs, train_loss, label="train_total_loss", color="#1f77b4", linewidth=2)
        ax_loss.plot(epochs, val_loss, label="val_total_loss", color="#d62728", linewidth=2)
        if not np.isnan(train_primary_loss).all():
            ax_loss.plot(epochs, train_primary_loss, label="train_primary_loss", color="#17becf", alpha=0.8)
        if not np.isnan(val_primary_loss).all():
            ax_loss.plot(epochs, val_primary_loss, label="val_primary_loss", color="#ff7f0e", alpha=0.8)
        ax_loss.legend(loc="best", fontsize=8)
    else:
        ax_loss.text(0.5, 0.5, "Waiting for first epoch", ha="center", va="center", transform=ax_loss.transAxes)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.2)

    if has_epoch_history:
        ax_metric.plot(epochs, val_bal_acc, label="val_balanced_accuracy", color="#2ca02c", linewidth=2)
        if not np.isnan(val_dir_hit).all():
            ax_metric.plot(epochs, val_dir_hit, label="val_directional_hit_rate", color="#9467bd", linewidth=2)
        ax_metric.legend(loc="best", fontsize=8)
    else:
        ax_metric.text(0.5, 0.5, "Validation metrics pending", ha="center", va="center", transform=ax_metric.transAxes)
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_ylim(0.0, 1.0)
    ax_metric.set_title("Validation Metrics")
    ax_metric.grid(alpha=0.2)

    if has_epoch_history and np.isfinite(learning_rate).any():
        ax_lr.plot(epochs, learning_rate, color="#8c564b", linewidth=2)
        ax_lr.set_yscale("log")
    else:
        ax_lr.text(0.5, 0.5, "LR schedule pending", ha="center", va="center", transform=ax_lr.transAxes)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("LR Schedule")
    ax_lr.grid(alpha=0.2)

    if predicted_share_keys:
        palette = ["#d62728", "#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e"]
        for index, label in enumerate(predicted_share_keys):
            shares = [item.get(f"val_pred_share_{label}", np.nan) for item in history]
            ax_share.plot(
                epochs,
                shares,
                label=f"pred_{label}",
                color=palette[index % len(palette)],
                linewidth=2,
            )
        ax_share.set_ylim(0.0, 1.0)
        ax_share.legend(loc="best", fontsize=8)
    else:
        ax_share.text(0.5, 0.5, "No class-share data", ha="center", va="center", transform=ax_share.transAxes)
    ax_share.set_xlabel("Epoch")
    ax_share.set_ylabel("Share")
    ax_share.set_title("Validation Predicted Class Mix")
    ax_share.grid(alpha=0.2)

    if step_history:
        global_steps = [item["global_step"] for item in step_history]
        step_loss = [item.get("running_loss", np.nan) for item in step_history]
        step_primary = [item.get("running_primary_loss", np.nan) for item in step_history]
        step_throughput = [item.get("samples_per_second", np.nan) for item in step_history]
        ax_step.plot(global_steps, step_loss, label="step_loss", color="#1f77b4", linewidth=2)
        if not np.isnan(step_primary).all():
            ax_step.plot(global_steps, step_primary, label="step_primary_loss", color="#ff7f0e", linewidth=1.5)
        ax_step_rate = ax_step.twinx()
        if not np.isnan(step_throughput).all():
            ax_step_rate.plot(
                global_steps,
                step_throughput,
                label="samples_per_second",
                color="#2ca02c",
                linewidth=1.5,
            )
        ax_step_rate.set_ylabel("Samples / Second")
        lines, labels = ax_step.get_legend_handles_labels()
        rate_lines, rate_labels = ax_step_rate.get_legend_handles_labels()
        ax_step.legend(lines + rate_lines, labels + rate_labels, loc="best", fontsize=8)
    else:
        ax_step.text(0.5, 0.5, "Step metrics pending", ha="center", va="center", transform=ax_step.transAxes)
    ax_step.set_xlabel("Global Step")
    ax_step.set_ylabel("Loss")
    ax_step.set_title("Step Loss And Throughput")
    ax_step.grid(alpha=0.2)

    if telemetry_history:
        minutes = [item.get("elapsed_seconds", 0.0) / 60.0 for item in telemetry_history]
        gpu_util = [item.get("gpu_utilization_pct", np.nan) for item in telemetry_history]
        temperature_c = [item.get("temperature_c", np.nan) for item in telemetry_history]
        power_pct = [
            (item.get("power_draw_w", np.nan) / item.get("power_limit_w", np.nan) * 100.0)
            if item.get("power_limit_w", 0.0)
            else np.nan
            for item in telemetry_history
        ]
        graphics_clock_mhz = [item.get("graphics_clock_mhz", np.nan) for item in telemetry_history]
        sm_clock_mhz = [item.get("sm_clock_mhz", np.nan) for item in telemetry_history]
        ax_gpu.plot(minutes, gpu_util, label="gpu_util_%", color="#2ca02c", linewidth=2)
        if not np.isnan(temperature_c).all():
            ax_gpu.plot(minutes, temperature_c, label="temperature_c", color="#ff7f0e", linewidth=1.5)
        if not np.isnan(power_pct).all():
            ax_gpu.plot(minutes, power_pct, label="power_%_of_limit", color="#d62728", linewidth=1.5, linestyle="--")
        ax_gpu.set_ylim(0.0, 100.0)
        ax_gpu.set_xlabel("Elapsed Minutes")
        ax_gpu.set_ylabel("Util / Temp / Power %")
        ax_gpu_clock = ax_gpu.twinx()
        if not np.isnan(graphics_clock_mhz).all():
            ax_gpu_clock.plot(minutes, graphics_clock_mhz, label="graphics_clock_mhz", color="#1f77b4", linewidth=2)
        if not np.isnan(sm_clock_mhz).all():
            ax_gpu_clock.plot(minutes, sm_clock_mhz, label="sm_clock_mhz", color="#9467bd", linestyle="--", linewidth=1.5)
        ax_gpu_clock.set_ylabel("Clock (MHz)")
        lines, labels = ax_gpu.get_legend_handles_labels()
        clock_lines, clock_labels = ax_gpu_clock.get_legend_handles_labels()
        ax_gpu.legend(lines + clock_lines, labels + clock_labels, loc="best", fontsize=8)
    else:
        ax_gpu.text(0.5, 0.5, "GPU telemetry pending", ha="center", va="center", transform=ax_gpu.transAxes)
        ax_gpu.set_xlabel("Elapsed Minutes")
        ax_gpu.set_ylabel("Util / Temp / Power %")
    ax_gpu.set_title("GPU Compute Telemetry")
    ax_gpu.grid(alpha=0.2)

    fig.suptitle("Training Progress", fontsize=14)
    fig.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """Save a labeled confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_experiment_comparison(
    rows: list[dict[str, float | int | str | None]],
    output_path: str | Path,
) -> Path:
    """Save a multi-run comparison chart for sweep-style experiments."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No completed runs yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(target, dpi=150)
        plt.close(fig)
        return target

    names = [str(row["name"]) for row in rows]
    positions = np.arange(len(rows))
    val_log_loss = [float(row.get("validation_log_loss", np.nan)) for row in rows]
    test_log_loss = [float(row.get("test_log_loss", np.nan)) for row in rows]
    val_bal_acc = [float(row.get("validation_balanced_accuracy", np.nan)) for row in rows]
    test_bal_acc = [float(row.get("test_balanced_accuracy", np.nan)) for row in rows]
    val_dir_hit = [float(row.get("validation_directional_hit_rate", np.nan)) for row in rows]
    test_dir_hit = [float(row.get("test_directional_hit_rate", np.nan)) for row in rows]
    params_millions = [float(row.get("model_parameter_count", np.nan)) / 1_000_000.0 for row in rows]
    best_epoch = [float(row.get("best_epoch", np.nan)) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    width = 0.35

    ax = axes[0, 0]
    ax.bar(positions - width / 2, val_log_loss, width=width, label="validation", color="#d62728")
    ax.bar(positions + width / 2, test_log_loss, width=width, label="test", color="#ff9896")
    ax.set_title("Log Loss")
    ax.set_ylabel("Lower is better")
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.bar(positions - width / 2, val_bal_acc, width=width, label="validation", color="#2ca02c")
    ax.bar(positions + width / 2, test_bal_acc, width=width, label="test", color="#98df8a")
    ax.set_title("Balanced Accuracy")
    ax.set_ylabel("Higher is better")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.bar(positions - width / 2, val_dir_hit, width=width, label="validation", color="#1f77b4")
    ax.bar(positions + width / 2, test_dir_hit, width=width, label="test", color="#aec7e8")
    ax.set_title("Directional Hit Rate")
    ax.set_ylabel("Higher is better")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.bar(positions, params_millions, color="#9467bd", alpha=0.8, label="params (M)")
    ax.set_title("Model Size And Best Epoch")
    ax.set_ylabel("Parameters (millions)")
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    epoch_axis = ax.twinx()
    epoch_axis.plot(positions, best_epoch, color="#ff7f0e", marker="o", linewidth=2, label="best_epoch")
    epoch_axis.set_ylabel("Best Epoch")
    lines, labels = ax.get_legend_handles_labels()
    epoch_lines, epoch_labels = epoch_axis.get_legend_handles_labels()
    ax.legend(lines + epoch_lines, labels + epoch_labels, loc="best")

    fig.suptitle("Architecture Sweep Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target
