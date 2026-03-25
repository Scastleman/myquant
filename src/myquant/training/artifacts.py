from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json


ARTIFACTS_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "runs"


def create_run_dir(prefix: str = "baseline") -> Path:
    """Create a timestamped run directory under artifacts/runs."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ARTIFACTS_ROOT / f"{prefix}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(payload: dict, path: str | Path) -> Path:
    """Write JSON with readable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
