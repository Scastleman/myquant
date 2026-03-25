from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from myquant.config import load_project_config
from myquant.data.io import PROCESSED_DATASET_PATH, read_parquet

from .artifacts import create_run_dir, write_json
from .baselines import build_baseline_models
from .evaluation import evaluate_classifier_predictions


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns safe for baseline training."""
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


def _subset(frame: pd.DataFrame, split_name: str, feature_columns: list[str], target_column: str):
    subset = frame.loc[frame["split"] == split_name].copy()
    return subset, subset.loc[:, feature_columns], subset[target_column]


def _prediction_frame(
    subset: pd.DataFrame,
    predicted_labels: pd.Series,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    base = subset.loc[
        :,
        [
            "date",
            "split",
            "target_label_5d",
            "target_ret_5d",
            "target_ret_1d",
            "vix_abs_10pct_flag",
            "vix_abs_20pct_flag",
        ],
    ].copy()
    base["predicted_label"] = predicted_labels.values
    return pd.concat([base, probabilities], axis=1)


def _slice_metrics(predictions: pd.DataFrame, event_column: str) -> dict[str, dict[str, float] | int]:
    event_subset = predictions.loc[predictions[event_column] == 1]
    metrics = {"row_count": int(len(event_subset))}
    if event_subset.empty:
        return metrics

    probability_columns = [column for column in predictions.columns if column.startswith("proba_")]
    classes = [column.removeprefix("proba_") for column in probability_columns]
    metrics["scores"] = evaluate_classifier_predictions(
        event_subset["target_label_5d"],
        event_subset["predicted_label"],
        event_subset.loc[:, probability_columns].to_numpy(),
        classes=classes,
    )
    metrics["scores"]["average_signal_strength"] = float(
        (event_subset.get("proba_up", 0.0) - event_subset.get("proba_down", 0.0)).mean()
    )
    return metrics


def train_baselines(dataset: pd.DataFrame, run_dir: Path) -> dict:
    """Train baseline models and save metrics, predictions, and fitted artifacts."""
    feature_columns = get_feature_columns(dataset)
    target_column = "target_label_5d"
    train_frame, x_train, y_train = _subset(dataset, "train", feature_columns, target_column)
    validation_frame, x_validation, y_validation = _subset(dataset, "validation", feature_columns, target_column)
    test_frame, x_test, y_test = _subset(dataset, "test", feature_columns, target_column)

    del train_frame  # not used again directly

    summary: dict[str, dict] = {
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "models": {},
    }

    models = build_baseline_models()
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        metrics_by_split: dict[str, dict] = {}
        for split_name, split_frame, x_split, y_split in (
            ("validation", validation_frame, x_validation, y_validation),
            ("test", test_frame, x_test, y_test),
        ):
            predicted_labels = pd.Series(model.predict(x_split), index=split_frame.index)
            probabilities = model.predict_proba(x_split)
            probability_columns = [f"proba_{label}" for label in model.classes_]
            probability_frame = pd.DataFrame(probabilities, columns=probability_columns, index=split_frame.index)
            predictions = _prediction_frame(split_frame, predicted_labels, probability_frame)

            split_metrics = evaluate_classifier_predictions(
                y_split,
                predicted_labels,
                probabilities,
                classes=model.classes_,
            )
            split_metrics["average_signal_strength"] = float(
                (predictions.get("proba_up", 0.0) - predictions.get("proba_down", 0.0)).mean()
            )
            split_metrics["event_slices"] = {
                "vix_abs_10pct_flag": _slice_metrics(predictions, "vix_abs_10pct_flag"),
                "vix_abs_20pct_flag": _slice_metrics(predictions, "vix_abs_20pct_flag"),
            }
            metrics_by_split[split_name] = split_metrics

            predictions.to_parquet(model_dir / f"{split_name}_predictions.parquet", index=False)

        joblib.dump(model, model_dir / "model.joblib")
        write_json(metrics_by_split, model_dir / "metrics.json")
        summary["models"][model_name] = metrics_by_split

    best_model_name = min(
        summary["models"],
        key=lambda name: summary["models"][name]["validation"]["log_loss"],
    )
    summary["best_model_by_validation_log_loss"] = best_model_name
    write_json(summary, run_dir / "summary.json")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the phase-1 baseline model suite.")
    parser.add_argument(
        "--config",
        default="configs/project.toml",
        help="Path to the project TOML config.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROCESSED_DATASET_PATH),
        help="Processed dataset parquet path.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _ = load_project_config(args.config)
    dataset = read_parquet(args.dataset_path)
    run_dir = create_run_dir(prefix="baseline")
    summary = train_baselines(dataset=dataset, run_dir=run_dir)

    print(f"Saved baseline run to {run_dir}")
    print(f"Best model by validation log loss: {summary['best_model_by_validation_log_loss']}")
    for model_name, metrics in summary["models"].items():
        val = metrics["validation"]
        test = metrics["test"]
        print(
            f"{model_name}: "
            f"val_log_loss={val['log_loss']:.4f}, val_bal_acc={val['balanced_accuracy']:.4f}, "
            f"test_log_loss={test['log_loss']:.4f}, test_bal_acc={test['balanced_accuracy']:.4f}",
        )


if __name__ == "__main__":
    main()
