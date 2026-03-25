#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    category: str
    status: str
    epochs_target: int
    epochs_completed: int
    best_epoch: int
    best_image_auroc: float
    final_image_auroc: float
    max_pixel_iou: float
    final_pixel_iou: float
    stability: str


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_category(run_dir: Path, config_payload: dict | None = None) -> str:
    if config_payload and str(config_payload.get("category", "")).strip():
        return str(config_payload["category"])
    suffix = run_dir.name.split("_", maxsplit=2)
    if len(suffix) >= 3:
        return suffix[2]
    return run_dir.name


def _classify_stability(
    *,
    best_image_auroc: float,
    final_image_auroc: float,
    best_epoch: int,
    epochs_completed: int,
) -> str:
    gap = best_image_auroc - final_image_auroc
    if epochs_completed <= 0:
        return "no_history"
    if gap <= 0.01:
        return "stable"
    if best_epoch <= max(int(epochs_completed * 0.4), 1) and gap >= 0.05:
        return "early_peak_drift"
    if best_epoch >= max(int(epochs_completed * 0.8), 1):
        return "late_improving"
    return "moderate_drift"


def _load_history_rows(history_path: Path) -> list[dict[str, str]]:
    with history_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_run(run_dir: Path) -> RunSummary | None:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    history_path = run_dir / "logs" / "history.csv"

    summary_payload = _read_json(summary_path) if summary_path.exists() else {}
    config_payload = _read_json(config_path) if config_path.exists() else {}
    training_protocol = summary_payload.get("training_protocol", config_payload.get("training_protocol", ""))
    if training_protocol and training_protocol != "train_good_only":
        return None

    category = _infer_category(run_dir, config_payload)
    if not history_path.exists():
        return RunSummary(
            run_dir=run_dir.name,
            category=category,
            status="missing_history",
            epochs_target=int(summary_payload.get("epochs", config_payload.get("epochs", 0) or 0)),
            epochs_completed=0,
            best_epoch=0,
            best_image_auroc=0.0,
            final_image_auroc=0.0,
            max_pixel_iou=0.0,
            final_pixel_iou=0.0,
            stability="no_history",
        )

    rows = _load_history_rows(history_path)
    if not rows:
        return RunSummary(
            run_dir=run_dir.name,
            category=category,
            status="empty_history",
            epochs_target=int(summary_payload.get("epochs", config_payload.get("epochs", 0) or 0)),
            epochs_completed=0,
            best_epoch=0,
            best_image_auroc=0.0,
            final_image_auroc=0.0,
            max_pixel_iou=0.0,
            final_pixel_iou=0.0,
            stability="no_history",
        )

    best_row = max(rows, key=lambda row: float(row["image_auroc"]))
    final_row = rows[-1]
    epochs_target = int(summary_payload.get("epochs", config_payload.get("epochs", 0) or len(rows)))
    epochs_completed = len(rows)
    status = "completed" if summary_path.exists() and epochs_completed >= epochs_target else "incomplete"
    best_image_auroc = float(best_row["image_auroc"])
    final_image_auroc = float(final_row["image_auroc"])
    max_pixel_iou = max(float(row["pixel_iou"]) for row in rows)
    final_pixel_iou = float(final_row["pixel_iou"])
    best_epoch = int(best_row["epoch"])
    stability = _classify_stability(
        best_image_auroc=best_image_auroc,
        final_image_auroc=final_image_auroc,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
    )
    return RunSummary(
        run_dir=run_dir.name,
        category=category,
        status=status,
        epochs_target=epochs_target,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_image_auroc=best_image_auroc,
        final_image_auroc=final_image_auroc,
        max_pixel_iou=max_pixel_iou,
        final_pixel_iou=final_pixel_iou,
        stability=stability,
    )


def collect_runs(root: Path) -> list[RunSummary]:
    results: list[RunSummary] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = summarize_run(run_dir)
        if summary is not None:
            results.append(summary)
    return results


def latest_per_category(runs: Iterable[RunSummary]) -> list[RunSummary]:
    latest: dict[str, RunSummary] = {}
    for run in runs:
        current = latest.get(run.category)
        if current is None or run.run_dir > current.run_dir:
            latest[run.category] = run
    return sorted(latest.values(), key=lambda run: run.category)


def render_markdown_table(runs: list[RunSummary]) -> str:
    headers = [
        "Category",
        "Run",
        "Status",
        "Best AUROC",
        "Best Ep",
        "Final AUROC",
        "Max IoU",
        "Final IoU",
        "Stability",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for run in runs:
        rows.append(
            "| "
            + " | ".join(
                [
                    run.category,
                    run.run_dir,
                    run.status,
                    f"{run.best_image_auroc:.6f}",
                    str(run.best_epoch),
                    f"{run.final_image_auroc:.6f}",
                    f"{run.max_pixel_iou:.6f}",
                    f"{run.final_pixel_iou:.6f}",
                    run.stability,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def write_csv(path: Path, runs: list[RunSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(runs[0]).keys()) if runs else list(RunSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize category training runs.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--all-runs", action="store_true", help="Show every training run instead of latest per category.")
    parser.add_argument("--write-csv", default="", help="Optional path to write a CSV summary.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    runs = collect_runs(runs_root)
    selected = runs if args.all_runs else latest_per_category(runs)

    print(render_markdown_table(selected))
    if args.write_csv:
        write_csv(Path(args.write_csv), selected)


if __name__ == "__main__":
    main()
