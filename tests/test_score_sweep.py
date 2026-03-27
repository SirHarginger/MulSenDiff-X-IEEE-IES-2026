import csv
import json
from pathlib import Path

from src.experiments import score_sweep


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_descriptor_score_sweep_selects_best_dual_gate_variant(monkeypatch, tmp_path: Path) -> None:
    train_run = tmp_path / "runs" / "shared_run"
    checkpoint_path = train_run / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    selection_manifest = train_run / "manifests" / "selection_manifest.csv"
    official_manifest = train_run / "manifests" / "official_test_manifest.csv"
    selection_manifest.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest.write_text("sample_name\nselection_row\n", encoding="utf-8")
    official_manifest.write_text("sample_name\nofficial_row\n", encoding="utf-8")
    _write_json(
        train_run / "summary.json",
        {
            "best_checkpoint": str(checkpoint_path),
            "selected_categories": ["capsule", "screw"],
        },
    )
    _write_json(
        train_run / "config.json",
        {
            "selection_manifest_csv": str(selection_manifest),
            "official_test_manifest_csv": str(official_manifest),
            "selected_categories": ["capsule", "screw"],
        },
    )

    metrics_by_variant = {
        (0.25, 0.30): {
            "selection": {"macro": 0.80, "max_fp": 0.04, "max_anom": 0.42, "worst": 0.70},
            "official_test": {"macro": 0.79, "max_fp": 0.05, "max_anom": 0.40, "worst": 0.69},
        },
        (0.30, 0.20): {
            "selection": {"macro": 0.82, "max_fp": 0.04, "max_anom": 0.30, "worst": 0.74},
            "official_test": {"macro": 0.81, "max_fp": 0.05, "max_anom": 0.31, "worst": 0.73},
        },
        (0.35, 0.15): {
            "selection": {"macro": 0.81, "max_fp": 0.03, "max_anom": 0.28, "worst": 0.73},
            "official_test": {"macro": 0.80, "max_fp": 0.04, "max_anom": 0.29, "worst": 0.72},
        },
        (0.35, 0.10): {
            "selection": {"macro": 0.79, "max_fp": 0.04, "max_anom": 0.25, "worst": 0.68},
            "official_test": {"macro": 0.78, "max_fp": 0.05, "max_anom": 0.26, "worst": 0.67},
        },
    }

    def fake_evaluate_checkpoint(**kwargs):
        split_name = "official_test" if "official_test" in str(kwargs["eval_manifest_csv"]) else "selection"
        key = (
            round(float(kwargs["lambda_descriptor_spatial"]), 2),
            round(float(kwargs["lambda_descriptor_global"]), 2),
        )
        metrics = metrics_by_variant[key][split_name]
        run_dir = tmp_path / "sweep_eval_runs" / f"{kwargs['run_name']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "summary": {"run_dir": str(run_dir)},
            "eval_summary": {
                "image_auroc": metrics["macro"],
                "macro_image_auroc": metrics["macro"],
                "image_auprc": metrics["macro"] - 0.01,
                "macro_image_auprc": metrics["macro"] - 0.01,
                "good_score_q95": 1.2,
                "macro_good_score_q95": 1.2,
                "max_good_false_positive_rate": metrics["max_fp"],
                "max_good_false_positive_rate_category": "screw",
                "max_anomalous_normal_rate": metrics["max_anom"],
                "max_anomalous_normal_rate_category": "nut",
                "worst_image_auroc": metrics["worst"],
                "worst_image_auroc_category": "screw",
                "pixel_auroc": 0.65,
                "pixel_aupr": 0.60,
                "pixel_f1_max": 0.55,
                "aupro": 0.58,
                "per_category": {
                    "capsule": {
                        "good_false_positive_rate": min(metrics["max_fp"], 0.02),
                        "anomalous_normal_rate": min(metrics["max_anom"], 0.10),
                    },
                    "screw": {
                        "good_false_positive_rate": metrics["max_fp"],
                        "anomalous_normal_rate": metrics["max_anom"],
                    },
                },
            },
        }

    monkeypatch.setattr(score_sweep, "evaluate_checkpoint", fake_evaluate_checkpoint)

    result = score_sweep.run_descriptor_score_sweep(
        train_run_root=train_run,
        output_root=tmp_path / "sweep_runs",
        run_name="score_sweep",
        device="cpu",
        max_eval_batches=1,
        max_visualizations=1,
    )

    assert result["summary"]["recommended_variant"] == "moderate_rebalance"
    rows = list(
        csv.DictReader(
            Path(result["summary"]["score_sweep_csv"]).open("r", encoding="utf-8", newline="")
        )
    )
    assert len(rows) == 8
    recommended = json.loads(Path(result["summary"]["recommended_variant_json"]).read_text(encoding="utf-8"))
    assert recommended["recommended_variant"] == "moderate_rebalance"
    assert recommended["selection_run"]["selection_dual_constraint_satisfied"] is True
