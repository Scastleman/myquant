from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_curves(history: list[dict[str, float]], output_path: str | Path) -> Path:
    """Save training/validation loss and validation balanced accuracy curves."""
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_bal_acc = [item["val_balanced_accuracy"] for item in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_loss, label="train_loss", color="#1f77b4")
    ax1.plot(epochs, val_loss, label="val_loss", color="#d62728")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_bal_acc, label="val_balanced_accuracy", color="#2ca02c")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_ylim(0.0, 1.0)

    lines, labels = [], []
    for axis in (ax1, ax2):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax1.legend(lines, labels, loc="best")
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
