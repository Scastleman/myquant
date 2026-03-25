from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss


def _one_hot_encode(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    one_hot = np.zeros((len(labels), len(classes)), dtype=float)
    for row_idx, label in enumerate(labels):
        one_hot[row_idx, class_to_index[label]] = 1.0
    return one_hot


def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    """Compute the mean multiclass Brier score."""
    y_encoded = _one_hot_encode(y_true, classes)
    return float(np.mean(np.sum((y_proba - y_encoded) ** 2, axis=1)))


def directional_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    up_label: str = "up",
    down_label: str = "down",
) -> float:
    """Accuracy on non-flat rows, counting 'flat' predictions as misses."""
    mask = np.isin(y_true, [up_label, down_label])
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Top-label expected calibration error for multiclass classification."""
    confidences = np.max(y_proba, axis=1)
    correctness = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue

        bucket_accuracy = correctness[mask].mean()
        bucket_confidence = confidences[mask].mean()
        ece += (mask.mean()) * abs(bucket_accuracy - bucket_confidence)

    return float(ece)


def evaluate_classifier_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    y_proba: np.ndarray,
    classes: Iterable[str],
) -> dict[str, float]:
    """Evaluate baseline classifier outputs with the project's first metrics."""
    y_true_array = np.asarray(list(y_true))
    y_pred_array = np.asarray(list(y_pred))
    classes_array = np.asarray(list(classes))
    clipped_proba = np.clip(y_proba, 1e-12, 1.0)
    clipped_proba = clipped_proba / clipped_proba.sum(axis=1, keepdims=True)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "log_loss": float(log_loss(y_true_array, clipped_proba, labels=classes_array)),
        "brier_score": multiclass_brier_score(y_true_array, clipped_proba, classes_array),
        "directional_hit_rate": directional_hit_rate(y_true_array, y_pred_array),
        "ece": expected_calibration_error(y_true_array, y_pred_array, clipped_proba),
    }
