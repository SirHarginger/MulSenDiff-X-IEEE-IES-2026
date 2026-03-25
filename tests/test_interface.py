import json
from pathlib import Path

from app.dashboard_data import (
    default_app_sessions_root,
    filter_evidence_entries,
    find_available_evaluation_runs,
    find_available_training_runs,
    list_checkpoint_files,
    load_sample_artifacts,
    load_run_bundle,
    match_uploaded_rgb_name,
    resolve_repo_relative_path,
)
from app.gemini_client import save_gemini_generation
from src.data_loader import ProcessedSampleRecord
from src.explainer.evidence_builder import EvidencePackage
from src.explainer.retriever import RetrievedContextItem


def test_interface_run_discovery_and_loading(tmp_path: Path) -> None:
    training_run = tmp_path / "runs" / "20260320_capsule_train"
    (training_run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (training_run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    (training_run / "checkpoints" / "epoch_001.pt").write_bytes(b"checkpoint")
    (training_run / "config.json").write_text(json.dumps({"category": "capsule"}), encoding="utf-8")

    eval_run = tmp_path / "runs" / "20260320_capsule_eval"
    (eval_run / "evidence").mkdir(parents=True, exist_ok=True)
    (eval_run / "summary.json").write_text(
        json.dumps(
            {
                "category": "capsule",
                "checkpoint_path": "runs/20260320_capsule_train/checkpoints/best.pt",
                "selected_categories": ["all"],
            }
        ),
        encoding="utf-8",
    )
    (eval_run / "evidence" / "calibration.json").write_text(json.dumps({"q95": 0.3}), encoding="utf-8")
    (eval_run / "evidence" / "index.json").write_text(json.dumps([{"sample_id": "0"}]), encoding="utf-8")
    (eval_run / "metrics").mkdir(parents=True, exist_ok=True)
    (eval_run / "plots").mkdir(parents=True, exist_ok=True)
    (eval_run / "metrics" / "evaluation.json").write_text(json.dumps({"image_auroc": 0.9}), encoding="utf-8")
    (eval_run / "metrics" / "per_category.json").write_text(
        json.dumps({"capsule": {"image_auroc": 0.9}}),
        encoding="utf-8",
    )
    (eval_run / "metrics" / "preview_selection.json").write_text(
        json.dumps({"strategy": "shared_balanced_random"}),
        encoding="utf-8",
    )
    (eval_run / "metrics" / "image_score_data.json").write_text(
        json.dumps([{"score": 0.8, "label": 1, "category": "capsule"}]),
        encoding="utf-8",
    )
    (eval_run / "plots" / "evaluation_summary.png").write_bytes(b"png")

    training_runs = find_available_training_runs(tmp_path / "runs")
    evaluation_runs = find_available_evaluation_runs(tmp_path / "runs")

    assert len(training_runs) == 1
    assert training_runs[0].best_checkpoint_path.name == "best.pt"
    assert len(list_checkpoint_files(training_runs[0].root)) == 2

    assert len(evaluation_runs) == 1
    bundle = load_run_bundle(evaluation_runs[0].root)
    assert bundle["summary"]["category"] == "capsule"
    assert bundle["summary"]["selected_categories"] == ["capsule"]
    assert bundle["calibration"]["q95"] == 0.3
    assert bundle["evidence_index"][0]["sample_id"] == "0"
    assert bundle["evaluation"]["image_auroc"] == 0.9
    assert bundle["per_category"]["capsule"]["image_auroc"] == 0.9
    assert bundle["preview_selection"]["strategy"] == "shared_balanced_random"
    assert len(bundle["image_score_data"]) == 1
    assert "evaluation_summary.png" in bundle["plot_paths"]

    resolved = resolve_repo_relative_path("runs/example/file.json", tmp_path)
    assert resolved == tmp_path / "runs" / "example" / "file.json"


def test_sample_artifacts_filters_matching_and_gemini_save(tmp_path: Path) -> None:
    repo_root = tmp_path
    eval_run = repo_root / "runs" / "demo_eval"
    package_path = eval_run / "evidence" / "packages" / "capsule" / "crack" / "000.json"
    report_path = eval_run / "evidence" / "reports" / "capsule" / "crack" / "000.md"
    llm_bundle_path = eval_run / "evidence" / "llm" / "capsule" / "crack" / "000.json"
    visualization_path = eval_run / "predictions" / "eval" / "capsule_crack_000.png"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    llm_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_path.write_bytes(b"png")

    package = EvidencePackage(
        category="capsule",
        split="test",
        defect_label="crack",
        sample_id="000",
        is_anomalous=True,
        score_mode="noise_error",
        score_label="Noise Error",
        raw_score=1.25,
        severity_0_100=78.0,
        confidence_0_100=64.0,
        status="Suspicious",
        affected_area_pct=4.0,
        peak_anomaly_0_100=88.0,
        top_regions=[{"centroid_xy": [5, 6], "bbox_xyxy": [1, 2, 8, 9]}],
        rgb_observations=["Localized RGB anomaly."],
        thermal_observations=["Thermal support present."],
        geometric_observations=["Geometric support present."],
        cross_modal_support=["Cross-modal agreement is moderate."],
        evidence_breakdown={"score_basis_pct": 40.0, "thermal_pct": 20.0, "geometric_pct": 20.0, "crossmodal_pct": 20.0},
        confidence_notes=["Review manually."],
        global_descriptor_score=0.42,
        retrieved_context=[{"title": "capsule", "snippet": "capsule snippet", "source": "knowledge/capsule.md", "score": 0.9}],
        provenance={"source_eval_run": "runs/demo_eval"},
        source_paths={"visualization_path": str(visualization_path.relative_to(repo_root))},
    )
    package_path.write_text(json.dumps(package.to_dict()), encoding="utf-8")
    report_path.write_text("Saved report", encoding="utf-8")
    llm_bundle_path.write_text(
        json.dumps(
            {
                "prompt_payload": {"task": "industrial_anomaly_explanation"},
                "retrieved_context": package.retrieved_context,
                "rendered_markdown": "# Detection Statement\nSaved explanation",
            }
        ),
        encoding="utf-8",
    )

    entry = {
        "category": "capsule",
        "defect_label": "crack",
        "status": "Suspicious",
        "sample_id": "000",
        "package_path": str(package_path.relative_to(repo_root)),
        "report_path": str(report_path.relative_to(repo_root)),
        "llm_bundle_path": str(llm_bundle_path.relative_to(repo_root)),
    }

    artifacts = load_sample_artifacts(repo_root=repo_root, entry=entry)
    assert artifacts["report_text"] == "Saved report"
    assert artifacts["llm_markdown"].startswith("# Detection Statement")
    assert artifacts["visualization_path"] == visualization_path

    filtered = filter_evidence_entries([entry], category="capsule", defect_label="crack", status="Suspicious")
    assert filtered == [entry]
    assert filter_evidence_entries([entry], category="button_cell") == []

    records = [
        ProcessedSampleRecord(
            category="capsule",
            split="test",
            defect_label="crack",
            sample_id="000",
            sample_name="capsule__test__crack__000",
            sample_dir=str(repo_root / "data" / "processed" / "samples" / "capsule__test__crack__000"),
            is_anomalous=True,
            gt_mask_path="",
            rgb_source_path=str(repo_root / "data" / "raw" / "MulSen_AD" / "capsule" / "RGB" / "test" / "crack" / "0.png"),
            ir_source_path="",
            pc_source_path="",
        )
    ]
    assert match_uploaded_rgb_name("0.png", records) == records
    assert match_uploaded_rgb_name("capsule__test__crack__000.png", records) == records

    generation_paths = save_gemini_generation(
        output_root=default_app_sessions_root(repo_root),
        sample_slug="capsule_crack_000",
        source_eval_run=eval_run,
        source_evidence_path=package_path,
        package=package,
        retrieved_context=[
            RetrievedContextItem(
                title="capsule",
                snippet="capsule snippet",
                source="knowledge/capsule.md",
                score=0.9,
            ),
            {"title": "fallback", "snippet": "dict item", "source": "knowledge/other.md", "score": 0.2},
        ],
        prompt_payload={"task": "industrial_anomaly_explanation"},
        generation={"provider": "gemini", "model": "gemini-test", "markdown": "# Detection Statement\nGenerated", "response_payload": {}},
    )

    saved_json = json.loads(Path(generation_paths["json_path"]).read_text(encoding="utf-8"))
    assert Path(generation_paths["markdown_path"]).exists()
    assert saved_json["provider"] == "gemini"
    assert len(saved_json["retrieved_context"]) == 2
