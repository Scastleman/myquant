from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import webbrowser

from myquant.data.io import LARGE_UNIVERSE_PANEL_DATASET_PATH

from .artifacts import create_run_dir, write_json
from .plots import plot_experiment_comparison


@dataclass(frozen=True)
class SweepExperiment:
    name: str
    description: str
    overrides: dict[str, str | int | float | bool]


def _open_dashboard(path: Path) -> None:
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            webbrowser.open(path.as_uri(), new=2)
    except OSError:
        return


def _build_default_experiments() -> tuple[SweepExperiment, ...]:
    return (
        SweepExperiment(
            name="patchtst_core",
            description="Suggested default: 256 lookback, 16/8 patching, 4x128 backbone, RevIN, pre-norm.",
            overrides={},
        ),
        SweepExperiment(
            name="shorter_context",
            description="Nearby context ablation: shorter 192-step lookback with the same patching and width.",
            overrides={"lookback": 192},
        ),
        SweepExperiment(
            name="coarser_patches",
            description="Nearby tokenization ablation: 24/12 patching to smooth more local noise.",
            overrides={"patch_length": 24, "patch_stride": 12},
        ),
        SweepExperiment(
            name="lighter_backbone",
            description="Nearby capacity ablation: smaller 3-layer, 96-dim backbone with narrower FFN and regime head.",
            overrides={"d_model": 96, "num_layers": 3, "ff_dim": 192, "regime_dim": 24},
        ),
    )


def _build_regularized_refine_experiments() -> tuple[SweepExperiment, ...]:
    return (
        SweepExperiment(
            name="core_regularized",
            description="PatchTST core with stronger dropout, attention dropout, and weight decay.",
            overrides={
                "dropout": 0.25,
                "attention_dropout": 0.10,
                "weight_decay": 0.002,
            },
        ),
        SweepExperiment(
            name="core_regularized_sparse",
            description="PatchTST core with stronger regularization and less-overlapping windows.",
            overrides={
                "dropout": 0.25,
                "attention_dropout": 0.10,
                "weight_decay": 0.002,
                "window_stride": 12,
            },
        ),
        SweepExperiment(
            name="coarse_regularized",
            description="Coarser 24/12 patching with stronger dropout, attention dropout, and weight decay.",
            overrides={
                "patch_length": 24,
                "patch_stride": 12,
                "dropout": 0.25,
                "attention_dropout": 0.10,
                "weight_decay": 0.002,
            },
        ),
        SweepExperiment(
            name="coarse_regularized_sparse",
            description="Coarser 24/12 patching with stronger regularization and less-overlapping windows.",
            overrides={
                "patch_length": 24,
                "patch_stride": 12,
                "dropout": 0.25,
                "attention_dropout": 0.10,
                "weight_decay": 0.002,
                "window_stride": 12,
            },
        ),
    )


def _build_winner_ablation_experiments() -> tuple[SweepExperiment, ...]:
    base_overrides = {
        "patch_length": 24,
        "patch_stride": 12,
        "dropout": 0.25,
        "attention_dropout": 0.10,
        "weight_decay": 0.002,
        "window_stride": 12,
    }
    return (
        SweepExperiment(
            name="winner_baseline",
            description="Current best coarse sparse setup with auxiliary task and RevIN enabled.",
            overrides=dict(base_overrides),
        ),
        SweepExperiment(
            name="winner_no_aux",
            description="Current best coarse sparse setup with the auxiliary task removed.",
            overrides={
                **base_overrides,
                "aux_target_columns": "",
                "aux_loss_weight": 0.0,
            },
        ),
        SweepExperiment(
            name="winner_no_revin",
            description="Current best coarse sparse setup without RevIN.",
            overrides={
                **base_overrides,
                "use_revin": False,
            },
        ),
        SweepExperiment(
            name="winner_no_aux_no_revin",
            description="Current best coarse sparse setup with neither auxiliary task nor RevIN.",
            overrides={
                **base_overrides,
                "aux_target_columns": "",
                "aux_loss_weight": 0.0,
                "use_revin": False,
            },
        ),
    )


def _resolve_experiments(preset: str) -> tuple[SweepExperiment, ...]:
    if preset == "initial":
        return _build_default_experiments()
    if preset == "regularized_refine":
        return _build_regularized_refine_experiments()
    if preset == "winner_ablation":
        return _build_winner_ablation_experiments()
    raise ValueError(f"Unsupported preset: {preset}")


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _format_metric(value: float | int | str | None, *, precision: int = 4) -> str:
    if value is None:
        return "pending"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value}"
    return f"{value:.{precision}f}"


def _status_table_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<tr><td colspan='8'>No runs yet.</td></tr>"

    html_rows: list[str] = []
    for row in rows:
        dashboard_path = row.get("dashboard_path")
        summary_path = row.get("summary_path")
        dashboard_html = (
            f"<a href='{escape(str(dashboard_path))}'>dashboard</a>"
            if isinstance(dashboard_path, str)
            else "pending"
        )
        summary_html = (
            f"<a href='{escape(str(summary_path))}'>summary</a>"
            if isinstance(summary_path, str)
            else "pending"
        )
        html_rows.append(
            "<tr>"
            f"<td>{escape(str(row['name']))}</td>"
            f"<td>{escape(str(row['status']))}</td>"
            f"<td>{_format_metric(row.get('validation_log_loss'))}</td>"
            f"<td>{_format_metric(row.get('test_log_loss'))}</td>"
            f"<td>{_format_metric(row.get('validation_balanced_accuracy'))}</td>"
            f"<td>{_format_metric(row.get('test_balanced_accuracy'))}</td>"
            f"<td>{_format_metric(row.get('model_parameter_count'), precision=0)}</td>"
            f"<td>{dashboard_html} | {summary_html}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def _write_dashboard(
    *,
    dashboard_path: Path,
    sweep_dir: Path,
    status: dict[str, object],
    rows: list[dict[str, object]],
    comparison_plot_path: Path,
    refresh_seconds: int,
) -> None:
    current_run = status.get("current_run")
    if isinstance(current_run, dict):
        current_name = current_run.get("name", "pending")
        current_phase = current_run.get("phase", "starting")
        current_epoch = current_run.get("latest_epoch")
        total_epochs = current_run.get("total_epochs")
        current_step = current_run.get("latest_step")
        total_steps = current_run.get("total_steps")
        current_best = current_run.get("best_epoch_so_far")
        current_loss = current_run.get("latest_running_loss")
        current_lr = current_run.get("latest_learning_rate")
        current_speed = current_run.get("latest_samples_per_second")
        dashboard_link = current_run.get("dashboard_path")
        current_summary = (
            f"{escape(str(current_name))}: phase={escape(str(current_phase))}, "
            f"epoch={escape(str(current_epoch))}/{escape(str(total_epochs))}, "
            f"step={escape(str(current_step))}/{escape(str(total_steps))}, "
            f"best={escape(str(current_best))}, "
            f"loss={_format_metric(current_loss)}, lr={_format_metric(current_lr, precision=2)}, "
            f"samples/s={_format_metric(current_speed, precision=1)}"
        )
        current_link = (
            f"<a href='{escape(str(dashboard_link))}'>Open current run dashboard</a>"
            if isinstance(dashboard_link, str)
            else "Current run dashboard pending"
        )
    else:
        current_summary = "No active run."
        current_link = "Current run dashboard pending"

    comparison_href = f"{comparison_plot_path.name}?t={int(time.time())}"
    rows_html = _status_table_rows(rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>myquant Architecture Sweep</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f7f8fb; }}
    .panel {{ background: white; border: 1px solid #d9dee8; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; border: 1px solid #d9dee8; border-radius: 8px; background: white; }}
    .links {{ margin-top: 8px; }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Architecture Sweep</h1>
    <p>Auto-refreshing sweep dashboard for <code>{escape(str(sweep_dir))}</code>.</p>
    <p><strong>Status:</strong> {escape(str(status.get("phase", "starting")))}</p>
    <p><strong>Current run:</strong> {current_summary}</p>
    <p>{current_link}</p>
  </div>
  <div class="panel">
    <h2>Comparison Plot</h2>
    <img src="{escape(comparison_href)}" alt="Experiment comparison">
  </div>
  <div class="panel">
    <h2>Leaderboard</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Status</th>
          <th>Val Log Loss</th>
          <th>Test Log Loss</th>
          <th>Val Bal Acc</th>
          <th>Test Bal Acc</th>
          <th>Params</th>
          <th>Artifacts</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""
    dashboard_path.write_text(html, encoding="utf-8")


def _completed_row(
    *,
    name: str,
    description: str,
    run_dir: Path,
    summary: dict,
) -> dict[str, object]:
    validation = summary.get("validation", {})
    test = summary.get("test", {})
    return {
        "name": name,
        "description": description,
        "status": "completed",
        "run_dir": str(run_dir),
        "dashboard_path": str(run_dir / "dashboard.html"),
        "summary_path": str(run_dir / "summary.json"),
        "best_epoch": summary.get("best_epoch"),
        "model_parameter_count": summary.get("model_parameter_count"),
        "validation_log_loss": validation.get("log_loss"),
        "test_log_loss": test.get("log_loss"),
        "validation_balanced_accuracy": validation.get("balanced_accuracy"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "validation_directional_hit_rate": validation.get("directional_hit_rate"),
        "test_directional_hit_rate": test.get("directional_hit_rate"),
    }


def _pending_row(experiment: SweepExperiment, run_dir: Path) -> dict[str, object]:
    return {
        "name": experiment.name,
        "description": experiment.description,
        "status": "pending",
        "run_dir": str(run_dir),
        "dashboard_path": str(run_dir / "dashboard.html"),
        "summary_path": str(run_dir / "summary.json"),
    }


def _failed_row(experiment: SweepExperiment, run_dir: Path, return_code: int) -> dict[str, object]:
    row = _pending_row(experiment, run_dir)
    row["status"] = f"failed ({return_code})"
    return row


def _update_artifacts(
    *,
    sweep_dir: Path,
    rows: list[dict[str, object]],
    status: dict[str, object],
    refresh_seconds: int,
) -> None:
    leaderboard_rows = [row for row in rows if str(row.get("status")) == "completed"]
    comparison_rows = [
        {
            "name": row["name"],
            "validation_log_loss": row.get("validation_log_loss"),
            "test_log_loss": row.get("test_log_loss"),
            "validation_balanced_accuracy": row.get("validation_balanced_accuracy"),
            "test_balanced_accuracy": row.get("test_balanced_accuracy"),
            "validation_directional_hit_rate": row.get("validation_directional_hit_rate"),
            "test_directional_hit_rate": row.get("test_directional_hit_rate"),
            "model_parameter_count": row.get("model_parameter_count"),
            "best_epoch": row.get("best_epoch"),
        }
        for row in leaderboard_rows
    ]
    comparison_plot_path = plot_experiment_comparison(comparison_rows, sweep_dir / "comparison.png")
    write_json({"rows": rows}, sweep_dir / "leaderboard.json")
    write_json(status, sweep_dir / "sweep_status.json")
    _write_dashboard(
        dashboard_path=sweep_dir / "dashboard.html",
        sweep_dir=sweep_dir,
        status=status,
        rows=rows,
        comparison_plot_path=comparison_plot_path,
        refresh_seconds=refresh_seconds,
    )


def _base_args(args: argparse.Namespace) -> dict[str, str | int | float | bool]:
    return {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "loader_cache_mode": args.loader_cache_mode,
        "lookback": 256,
        "window_stride": 8,
        "patch_length": 16,
        "patch_stride": 8,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 4,
        "regime_dim": 32,
        "ff_dim": 256,
        "dropout": 0.2,
        "attention_dropout": 0.05,
        "learning_rate": 3e-4,
        "warmup_fraction": 0.1,
        "min_lr_ratio": 0.1,
        "weight_decay": 1e-3,
        "patience": args.patience,
        "amp_dtype": "auto",
        "accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "log_every_steps": args.log_every_steps,
        "primary_target_column": "target_label_5d",
        "aux_target_columns": "target_excess_label_5d",
        "aux_loss_weight": 0.15,
        "focus_ticker": args.focus_ticker,
        "use_revin": True,
        "no_open_dashboard": True,
    }


def _cli_args_from_config(config: dict[str, str | int | float | bool], run_dir: Path) -> list[str]:
    ordered_keys = [
        "dataset_path",
        "device",
        "epochs",
        "batch_size",
        "num_workers",
        "loader_cache_mode",
        "lookback",
        "window_stride",
        "patch_length",
        "patch_stride",
        "d_model",
        "n_heads",
        "num_layers",
        "regime_dim",
        "ff_dim",
        "dropout",
        "attention_dropout",
        "learning_rate",
        "warmup_fraction",
        "min_lr_ratio",
        "weight_decay",
        "patience",
        "amp_dtype",
        "accumulation_steps",
        "max_grad_norm",
        "log_every_steps",
        "primary_target_column",
        "aux_target_columns",
        "aux_loss_weight",
        "focus_ticker",
        "use_revin",
        "no_open_dashboard",
    ]
    flag_map = {
        "dataset_path": "--dataset-path",
        "device": "--device",
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "num_workers": "--num-workers",
        "loader_cache_mode": "--loader-cache-mode",
        "lookback": "--lookback",
        "window_stride": "--window-stride",
        "patch_length": "--patch-length",
        "patch_stride": "--patch-stride",
        "d_model": "--d-model",
        "n_heads": "--n-heads",
        "num_layers": "--num-layers",
        "regime_dim": "--regime-dim",
        "ff_dim": "--ff-dim",
        "dropout": "--dropout",
        "attention_dropout": "--attention-dropout",
        "learning_rate": "--learning-rate",
        "warmup_fraction": "--warmup-fraction",
        "min_lr_ratio": "--min-lr-ratio",
        "weight_decay": "--weight-decay",
        "patience": "--patience",
        "amp_dtype": "--amp-dtype",
        "accumulation_steps": "--accumulation-steps",
        "max_grad_norm": "--max-grad-norm",
        "log_every_steps": "--log-every-steps",
        "primary_target_column": "--primary-target-column",
        "aux_target_columns": "--aux-target-columns",
        "aux_loss_weight": "--aux-loss-weight",
        "focus_ticker": "--focus-ticker",
        "use_revin": "--use-revin",
        "no_open_dashboard": "--no-open-dashboard",
    }
    cli_args: list[str] = [
        sys.executable,
        "-m",
        "myquant.training.run_transformer",
        "--run-dir",
        str(run_dir),
    ]
    for key in ordered_keys:
        value = config[key]
        flag = flag_map[key]
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small PatchTST-style architecture sweep.")
    parser.add_argument(
        "--preset",
        choices=("initial", "regularized_refine", "winner_ablation"),
        default="initial",
        help="Named sweep preset to run.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(LARGE_UNIVERSE_PANEL_DATASET_PATH),
        help="Dataset parquet to evaluate. Defaults to the large-universe panel dataset.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--loader-cache-mode", choices=("auto", "ram", "disk"), default="ram")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--focus-ticker", default="SPY")
    parser.add_argument("--refresh-seconds", type=int, default=10)
    parser.add_argument(
        "--no-open-dashboard",
        action="store_true",
        help="Do not auto-open the sweep dashboard in a browser window.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    sweep_dir = create_run_dir(prefix=f"transformer-architecture-sweep-{args.preset}")
    experiments = _resolve_experiments(args.preset)
    rows: list[dict[str, object]] = []
    status: dict[str, object] = {
        "phase": "starting",
        "current_run": None,
        "sweep_dir": str(sweep_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "preset": args.preset,
        "experiments": [
            {"name": experiment.name, "description": experiment.description}
            for experiment in experiments
        ],
    }
    _update_artifacts(
        sweep_dir=sweep_dir,
        rows=rows,
        status=status,
        refresh_seconds=args.refresh_seconds,
    )
    dashboard_path = sweep_dir / "dashboard.html"
    if not args.no_open_dashboard:
        _open_dashboard(dashboard_path)

    base = _base_args(args)
    for experiment in experiments:
        run_dir = sweep_dir / experiment.name
        merged = dict(base)
        merged.update(experiment.overrides)
        pending = _pending_row(experiment, run_dir)
        rows = [row for row in rows if row["name"] != experiment.name]
        rows.append(pending)

        status["phase"] = "running"
        status["current_run"] = {
            "name": experiment.name,
            "description": experiment.description,
            "dashboard_path": str(run_dir / "dashboard.html"),
        }
        _update_artifacts(
            sweep_dir=sweep_dir,
            rows=rows,
            status=status,
            refresh_seconds=args.refresh_seconds,
        )

        command = _cli_args_from_config(merged, run_dir)
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3])
        while True:
            live_status = _safe_read_json(run_dir / "live_status.json")
            if live_status is not None:
                status["current_run"] = {
                    "name": experiment.name,
                    "description": experiment.description,
                    "dashboard_path": str(run_dir / "dashboard.html"),
                    **live_status,
                }
                _update_artifacts(
                    sweep_dir=sweep_dir,
                    rows=rows,
                    status=status,
                    refresh_seconds=args.refresh_seconds,
                )

            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(args.refresh_seconds)

        summary = _safe_read_json(run_dir / "summary.json")
        rows = [row for row in rows if row["name"] != experiment.name]
        if return_code == 0 and summary is not None:
            rows.append(
                _completed_row(
                    name=experiment.name,
                    description=experiment.description,
                    run_dir=run_dir,
                    summary=summary,
                )
            )
        else:
            rows.append(_failed_row(experiment, run_dir, return_code))
            status["phase"] = "failed"
            status["current_run"] = {
                "name": experiment.name,
                "description": experiment.description,
                "dashboard_path": str(run_dir / "dashboard.html"),
                "return_code": return_code,
            }
            _update_artifacts(
                sweep_dir=sweep_dir,
                rows=rows,
                status=status,
                refresh_seconds=args.refresh_seconds,
            )
            raise RuntimeError(f"Experiment {experiment.name} failed with exit code {return_code}")

        status["current_run"] = None
        _update_artifacts(
            sweep_dir=sweep_dir,
            rows=rows,
            status=status,
            refresh_seconds=args.refresh_seconds,
        )

    status["phase"] = "completed"
    status["current_run"] = None
    _update_artifacts(
        sweep_dir=sweep_dir,
        rows=rows,
        status=status,
        refresh_seconds=args.refresh_seconds,
    )

    print(f"Sweep completed: {sweep_dir}", flush=True)
    print(f"Dashboard: {dashboard_path}", flush=True)


if __name__ == "__main__":
    main()
