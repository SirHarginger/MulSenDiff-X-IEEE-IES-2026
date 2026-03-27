import json
from pathlib import Path

from src.experiments import detector_audit


def _write_eval_package(path: Path, *, category: str, defect_label: str, status: str, raw_score: float, affected_area_pct: float, global_descriptor_pct: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "category": category,
                "defect_label": defect_label,
                "status": status,
                "raw_score": raw_score,
                "affected_area_pct": affected_area_pct,
                "peak_anomaly_0_100": 90.0 if defect_label != "good" else 35.0,
                "global_descriptor_score": global_descriptor_pct / 25.0,
                "evidence_breakdown": {
                    "score_basis_pct": 20.0,
                    "thermal_pct": 10.0,
                    "geometric_pct": 10.0,
                    "crossmodal_pct": 5.0,
                    "global_descriptor_pct": global_descriptor_pct,
                },
                "source_paths": {
                    "rgb_path": f"data/processed/{category}_{defect_label}.png",
                    "source_rgb_path": f"data/raw/{category}_{defect_label}.png",
                    "visualization_path": f"runs/demo_eval/predictions/{category}_{defect_label}.png",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_build_audit_rows_and_review_queue(tmp_path: Path) -> None:
    eval_run = tmp_path / "runs" / "demo_eval"
    (eval_run / "evidence").mkdir(parents=True, exist_ok=True)
    (eval_run / "metrics").mkdir(parents=True, exist_ok=True)
    evidence_index = [
        {
            "category": "screw",
            "defect_label": "good",
            "sample_id": "000",
            "is_anomalous": False,
            "status": "Critical anomaly",
            "raw_score": 5.5,
            "severity_0_100": 100.0,
            "confidence_0_100": 95.0,
            "global_descriptor_score": 3.5,
            "package_path": "runs/demo_eval/evidence/packages/screw/good/000.json",
            "report_path": "runs/demo_eval/evidence/reports/screw/good/000.md",
        },
        {
            "category": "screw",
            "defect_label": "broken",
            "sample_id": "001",
            "is_anomalous": True,
            "status": "Normal",
            "raw_score": 0.7,
            "severity_0_100": 10.0,
            "confidence_0_100": 5.0,
            "global_descriptor_score": 0.1,
            "package_path": "runs/demo_eval/evidence/packages/screw/broken/001.json",
            "report_path": "runs/demo_eval/evidence/reports/screw/broken/001.md",
        },
    ]
    (eval_run / "evidence" / "index.json").write_text(json.dumps(evidence_index), encoding="utf-8")
    _write_eval_package(
        eval_run / "evidence" / "packages" / "screw" / "good" / "000.json",
        category="screw",
        defect_label="good",
        status="Critical anomaly",
        raw_score=5.5,
        affected_area_pct=28.0,
        global_descriptor_pct=80.0,
    )
    _write_eval_package(
        eval_run / "evidence" / "packages" / "screw" / "broken" / "001.json",
        category="screw",
        defect_label="broken",
        status="Normal",
        raw_score=0.7,
        affected_area_pct=2.0,
        global_descriptor_pct=25.0,
    )
    per_category = {
        "screw": {
            "image_auroc": 0.18,
            "image_auprc": 0.40,
            "mean_good_score": 6.0,
            "mean_anomalous_score": 4.1,
        }
    }

    rows = detector_audit.build_audit_rows(eval_run_root=eval_run, evidence_index=evidence_index)
    queue = detector_audit.build_manual_review_queue(audit_rows=rows)
    integrity = detector_audit.build_category_integrity_rows(audit_rows=rows, per_category_metrics=per_category)
    breakdown = detector_audit.build_evidence_breakdown_rows(eval_run_root=eval_run, evidence_index=evidence_index)

    assert len(rows) == 2
    assert len(queue) == 2
    assert queue[0]["review_label"] == ""
    assert integrity[0]["good_false_positive_rate"] == 1.0
    assert integrity[0]["anomalous_normal_rate"] == 1.0
    assert any(row["group"] == "good" for row in breakdown)
    assert any(row["group"] == "anomalous" for row in breakdown)


def test_run_detector_integrity_audit_writes_ablation_outputs(monkeypatch, tmp_path: Path) -> None:
    eval_run = tmp_path / "runs" / "demo_eval"
    (eval_run / "evidence").mkdir(parents=True, exist_ok=True)
    (eval_run / "metrics").mkdir(parents=True, exist_ok=True)
    (eval_run / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_path": "runs/demo_train/checkpoints/best.pt",
                "selected_categories": ["screw", "capsule"],
                "lambda_descriptor_global": 0.6,
                "lambda_descriptor_spatial": 0.25,
            }
        ),
        encoding="utf-8",
    )
    evidence_index = [
        {
            "category": "screw",
            "defect_label": "good",
            "sample_id": "000",
            "is_anomalous": False,
            "status": "Critical anomaly",
            "raw_score": 5.5,
            "severity_0_100": 100.0,
            "confidence_0_100": 95.0,
            "global_descriptor_score": 3.5,
            "package_path": "runs/demo_eval/evidence/packages/screw/good/000.json",
            "report_path": "runs/demo_eval/evidence/reports/screw/good/000.md",
        }
    ]
    (eval_run / "evidence" / "index.json").write_text(json.dumps(evidence_index), encoding="utf-8")
    (eval_run / "metrics" / "per_category.json").write_text(
        json.dumps(
            {
                "screw": {"image_auroc": 0.18, "image_auprc": 0.40, "mean_good_score": 6.0, "mean_anomalous_score": 4.1},
                "capsule": {"image_auroc": 0.82, "image_auprc": 0.90, "mean_good_score": 1.0, "mean_anomalous_score": 2.0},
            }
        ),
        encoding="utf-8",
    )
    (eval_run / "metrics" / "image_score_data.json").write_text(
        json.dumps(
            [
                {"category": "screw", "label": 0, "score": 5.5},
                {"category": "screw", "label": 1, "score": 3.0},
                {"category": "capsule", "label": 0, "score": 1.0},
                {"category": "capsule", "label": 1, "score": 2.0},
            ]
        ),
        encoding="utf-8",
    )
    _write_eval_package(
        eval_run / "evidence" / "packages" / "screw" / "good" / "000.json",
        category="screw",
        defect_label="good",
        status="Critical anomaly",
        raw_score=5.5,
        affected_area_pct=28.0,
        global_descriptor_pct=80.0,
    )

    def fake_evaluate_checkpoint(**kwargs):
        variant = str(kwargs.get("run_name", "variant"))
        run_dir = tmp_path / "ablations" / variant
        (run_dir / "evidence").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        variant_index = [
            {
                "category": "screw",
                "defect_label": "good",
                "sample_id": "000",
                "is_anomalous": False,
                "status": "Normal" if "no_global" in variant else "Critical anomaly",
                "raw_score": 1.0,
                "severity_0_100": 20.0,
                "confidence_0_100": 30.0,
                "global_descriptor_score": 0.0,
                "package_path": str((run_dir / "evidence" / "packages" / "screw" / "good" / "000.json").relative_to(tmp_path)),
                "report_path": "",
            }
        ]
        (run_dir / "evidence" / "index.json").write_text(json.dumps(variant_index), encoding="utf-8")
        (run_dir / "metrics" / "per_category.json").write_text(
            json.dumps({"screw": {"image_auroc": 0.5, "image_auprc": 0.5, "mean_good_score": 1.0, "mean_anomalous_score": 1.5}}),
            encoding="utf-8",
        )
        _write_eval_package(
            run_dir / "evidence" / "packages" / "screw" / "good" / "000.json",
            category="screw",
            defect_label="good",
            status="Normal" if "no_global" in variant else "Critical anomaly",
            raw_score=1.0,
            affected_area_pct=5.0,
            global_descriptor_pct=20.0,
        )
        return {
            "summary": {
                "run_dir": str(run_dir),
                "image_auroc": 0.5,
                "macro_image_auroc": 0.5,
                "image_auprc": 0.5,
                "macro_image_auprc": 0.5,
                "pixel_auroc": 0.4,
                "aupro": 0.3,
            }
        }

    monkeypatch.setattr(detector_audit, "evaluate_checkpoint", fake_evaluate_checkpoint)

    result = detector_audit.run_detector_integrity_audit(
        eval_run_root=eval_run,
        checkpoint_path=tmp_path / "runs" / "demo_train" / "checkpoints" / "best.pt",
        data_root=tmp_path / "data",
        output_root=tmp_path / "audit_runs",
        device="cpu",
    )

    summary = result["summary"]
    assert Path(summary["audit_run_dir"]).exists()
    assert Path(summary["audit_table_path"]).exists()
    assert Path(summary["manual_adjudication_queue_path"]).exists()
    assert Path(summary["category_integrity_path"]).exists()
    assert Path(summary["evidence_breakdown_path"]).exists()
    assert Path(summary["ablation_summary_path"]).exists()
    assert Path(summary["ablation_runs_path"]).exists()
