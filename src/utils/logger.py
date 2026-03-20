from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    plots: Path
    metrics: Path
    predictions: Path
    logs: Path


def create_run_dir(
    *,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    category: str | None = None,
) -> RunPaths:
    output_root = Path(output_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug_parts = [timestamp]
    if category:
        slug_parts.append(category)
    if run_name:
        slug_parts.append(run_name)
    root = output_root / "_".join(slug_parts)
    checkpoints = root / "checkpoints"
    plots = root / "plots"
    metrics = root / "metrics"
    predictions = root / "predictions"
    logs = root / "logs"
    for path in [checkpoints, plots, metrics, predictions, logs]:
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root, checkpoints=checkpoints, plots=plots, metrics=metrics, predictions=predictions, logs=logs)


def write_json(path: Path | str, payload: Dict[str, Any] | Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_history_csv(path: Path | str, rows: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
