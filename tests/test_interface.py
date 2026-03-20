import json
from pathlib import Path

from app.dashboard_data import (
    find_available_evaluation_runs,
    find_available_training_runs,
    list_checkpoint_files,
    load_run_bundle,
    resolve_repo_relative_path,
)


def test_interface_run_discovery_and_loading(tmp_path: Path) -> None:
    training_run = tmp_path / "runs" / "20260320_capsule_train"
    (training_run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (training_run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    (training_run / "checkpoints" / "epoch_001.pt").write_bytes(b"checkpoint")
    (training_run / "config.json").write_text(json.dumps({"category": "capsule"}), encoding="utf-8")

    eval_run = tmp_path / "runs" / "20260320_capsule_eval"
    (eval_run / "evidence").mkdir(parents=True, exist_ok=True)
    (eval_run / "summary.json").write_text(json.dumps({"category": "capsule"}), encoding="utf-8")
    (eval_run / "evidence" / "calibration.json").write_text(json.dumps({"q95": 0.3}), encoding="utf-8")
    (eval_run / "evidence" / "index.json").write_text(json.dumps([{"sample_id": "0"}]), encoding="utf-8")

    training_runs = find_available_training_runs(tmp_path / "runs")
    evaluation_runs = find_available_evaluation_runs(tmp_path / "runs")

    assert len(training_runs) == 1
    assert training_runs[0].best_checkpoint_path.name == "best.pt"
    assert len(list_checkpoint_files(training_runs[0].root)) == 2

    assert len(evaluation_runs) == 1
    bundle = load_run_bundle(evaluation_runs[0].root)
    assert bundle["summary"]["category"] == "capsule"
    assert bundle["calibration"]["q95"] == 0.3
    assert bundle["evidence_index"][0]["sample_id"] == "0"

    resolved = resolve_repo_relative_path("runs/example/file.json", tmp_path)
    assert resolved == tmp_path / "runs" / "example" / "file.json"
