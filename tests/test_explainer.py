import json
from pathlib import Path

import torch

from app.inference_runtime import load_shared_inference_session, run_known_sample_inference
from src.data_loader import DescriptorConditioningDataset, build_training_manifests, collate_model_samples
from src.data_loader import select_processed_sample_records
from src.explainer.evidence_builder import build_evidence_package
from src.explainer.report import render_templated_explanation
from src.explainer.retriever import retrieve_context_for_evidence
from src.inference.anomaly_scorer import score_batch
from src.inference.localization import (
    apply_localization_calibration,
    estimate_object_mask,
    fit_localization_calibration,
    normalize_anomaly_map,
    threshold_anomaly_map,
)
from src.inference.quantification import fit_masi_calibration
from src.models.llm_explainer import build_llm_explanation, build_prompt_payload, save_explanation_bundle
from src.models.mulsendiffx import MulSenDiffX
from src.training.train_diffusion import evaluate_checkpoint, fit_global_descriptor_reference, train_model
from tests.test_data_loader_runtime import _build_descriptor_ready_dataset


def test_build_evidence_package_and_report_runtime(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    manifests = build_training_manifests(data_root=root, categories=["capsule"])

    train_dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        target_size=(16, 16),
        global_stats_path=manifests["global_stats_json"],
    )
    eval_dataset = DescriptorConditioningDataset(
        manifests["eval_csv"],
        target_size=(16, 16),
        global_stats_path=manifests["global_stats_json"],
    )

    calibration_samples = [train_dataset[0], train_dataset[1]]
    eval_sample = eval_dataset[0]
    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=train_dataset.descriptor_channels,
        global_dim=train_dataset.global_dim,
        base_channels=8,
        global_embedding_dim=16,
        time_embedding_dim=16,
        attention_heads=2,
        diffusion_steps=50,
    )

    calibration_loader = torch.utils.data.DataLoader(calibration_samples, batch_size=1, collate_fn=collate_model_samples)
    global_calibration = fit_global_descriptor_reference(calibration_loader, category="capsule")

    calibration_scores = []
    for sample in calibration_samples:
        scored = score_batch(
            model,
            sample.x_rgb.unsqueeze(0),
            sample.conditioning.descriptor_maps.unsqueeze(0),
            sample.conditioning.support_maps.unsqueeze(0),
            sample.conditioning.global_vector.unsqueeze(0),
            timestep=10,
            global_descriptor_weight=0.1,
            global_descriptor_calibration=global_calibration,
        )
        calibration_scores.append(scored.anomaly_score.detach().cpu())
    calibration = fit_masi_calibration(torch.cat(calibration_scores), category="capsule", score_mode="noise_error")

    scored_eval = score_batch(
        model,
        eval_sample.x_rgb.unsqueeze(0),
        eval_sample.conditioning.descriptor_maps.unsqueeze(0),
        eval_sample.conditioning.support_maps.unsqueeze(0),
        eval_sample.conditioning.global_vector.unsqueeze(0),
        timestep=10,
        global_descriptor_weight=0.1,
        global_descriptor_calibration=global_calibration,
    )
    normalized_map = normalize_anomaly_map(scored_eval.anomaly_map.detach().cpu())
    object_mask = estimate_object_mask(eval_sample.x_rgb.unsqueeze(0), threshold=0.02)
    predicted_mask = threshold_anomaly_map(normalized_map, percentile=0.9, object_mask=object_mask, normalized=True)

    package = build_evidence_package(
        category=eval_sample.category,
        split=eval_sample.split,
        defect_label=eval_sample.defect_label,
        sample_id=eval_sample.sample_id,
        is_anomalous=eval_sample.is_anomalous,
        score_mode="noise_error",
        score_label="Noise Error",
        raw_score=float(scored_eval.anomaly_score[0].item()),
        calibration=calibration,
        normalized_anomaly_map=normalized_map[0],
        score_basis_map=scored_eval.score_basis_map[0].detach().cpu(),
        descriptor_support=scored_eval.descriptor_support[0].detach().cpu(),
        object_mask=object_mask[0].detach().cpu(),
        predicted_mask=predicted_mask[0].detach().cpu(),
        descriptor_maps=eval_sample.conditioning.descriptor_maps.detach().cpu(),
        support_maps=eval_sample.conditioning.support_maps.detach().cpu(),
        global_descriptor_score=float(scored_eval.global_descriptor_score[0].item()),
        provenance={"test": "runtime"},
        source_paths={
            "rgb_path": eval_sample.rgb_path,
            "gt_mask_path": eval_sample.gt_mask_path,
            "source_rgb_path": eval_sample.source_rgb_path,
        },
    )
    report = render_templated_explanation(package)

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "capsule_thermal.md").write_text(
        "Capsule thermal hotspot defects often correlate with localized heat concentration and roughness.",
        encoding="utf-8",
    )
    retrieved = retrieve_context_for_evidence(package, knowledge_base_root=knowledge_root, top_k=1)
    prompt_payload = build_prompt_payload(package, retrieved)
    explanation = build_llm_explanation(package, retrieved_context=retrieved, prompt_payload=prompt_payload)
    bundle_path = tmp_path / "llm_bundle.json"
    save_explanation_bundle(package, retrieved, prompt_payload, explanation, bundle_path)

    assert 0.0 <= package.severity_0_100 <= 100.0
    assert 0.0 <= package.confidence_0_100 <= 100.0
    assert package.top_regions
    assert abs(sum(package.evidence_breakdown.values()) - 100.0) < 1.0
    assert "Status:" in report
    assert "Observations:" in report
    assert retrieved
    assert bundle_path.exists()


def test_evaluate_checkpoint_writes_evidence_outputs(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    knowledge_root = root / "knowledge"
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "capsule_guidance.md").write_text(
        "Capsule anomalies with thermal and geometric agreement should be inspected for localized defects.",
        encoding="utf-8",
    )

    train_result = train_model(
        data_root=root,
        category="capsule",
        epochs=1,
        batch_size=1,
        target_size=(16, 16),
        device="cpu",
        output_root=root / "runs",
        max_train_batches=1,
        max_eval_batches=1,
        max_visualizations=1,
        base_channels=8,
        global_embedding_dim=16,
        time_embedding_dim=16,
        attention_heads=2,
        diffusion_steps=50,
    )
    best_checkpoint = Path(train_result["summary"]["best_checkpoint"])

    eval_result = evaluate_checkpoint(
        checkpoint_path=best_checkpoint,
        data_root=root,
        category="capsule",
        batch_size=1,
        target_size=(16, 16),
        device="cpu",
        output_root=root / "eval_runs",
        max_eval_batches=1,
        max_visualizations=1,
        knowledge_base_root=knowledge_root,
        enable_llm_explanations=True,
    )

    summary = eval_result["summary"]
    assert summary["evidence_packages_saved"] >= 1
    assert summary["llm_explanations_saved"] >= 1
    assert Path(summary["calibration_path"]).exists()
    assert Path(summary["localization_calibration_path"]).exists()
    assert Path(summary["evidence_index_path"]).exists()

    evidence_index = json.loads(Path(summary["evidence_index_path"]).read_text(encoding="utf-8"))
    assert len(evidence_index) >= 1
    assert Path(evidence_index[0]["package_path"]).exists()
    assert Path(evidence_index[0]["report_path"]).exists()
    assert Path(evidence_index[0]["llm_bundle_path"]).exists()

    inference_session = load_shared_inference_session(
        repo_root=root,
        eval_run_root=summary["run_dir"],
    )
    known_records = select_processed_sample_records(data_root=root, categories=["capsule"])
    live_result = run_known_sample_inference(
        inference_session,
        known_records[0],
        output_root=root / "app_sessions",
        knowledge_base_root=knowledge_root,
    )
    assert Path(live_result["package_path"]).exists()
    assert Path(live_result["retrieved_context_path"]).exists()
    assert Path(live_result["panel_path"]).exists()
    assert live_result["package"].category == "capsule"
    assert live_result["package"].provenance["localization_method"]


def test_localization_calibration_filters_small_components() -> None:
    calibration = fit_localization_calibration(
        torch.tensor([0.02, 0.03, 0.04, 0.05, 0.06], dtype=torch.float32),
        category="capsule",
        object_pixel_counts=torch.tensor([16.0, 16.0], dtype=torch.float32),
        quantile=0.8,
        min_region_area_fraction=0.0,
        min_region_pixels_floor=2,
        sample_count=2,
    )
    anomaly_map = torch.tensor(
        [[[[0.01, 0.06, 0.01, 0.01], [0.01, 0.07, 0.08, 0.01], [0.01, 0.01, 0.01, 0.09], [0.01, 0.01, 0.01, 0.01]]]],
        dtype=torch.float32,
    )
    object_mask = torch.ones_like(anomaly_map)
    predicted = apply_localization_calibration(
        anomaly_map,
        object_mask=object_mask,
        calibration=calibration,
        normalized=False,
    )

    assert calibration.min_region_pixels == 2
    assert int(predicted.sum().item()) == 3
