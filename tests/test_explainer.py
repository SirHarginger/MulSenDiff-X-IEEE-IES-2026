import json
from pathlib import Path

import torch

from src.data_loader import DescriptorConditioningDataset, build_training_manifests
from src.explainer.evidence_builder import build_evidence_package
from src.explainer.report import render_templated_explanation
from src.inference.anomaly_scorer import score_batch
from src.inference.localization import estimate_object_mask, normalize_anomaly_map, threshold_anomaly_map
from src.inference.quantification import fit_masi_calibration
from src.models.mulsendiffx import MulSenDiffX
from src.training.train_diffusion import evaluate_checkpoint, train_model
from tests.test_data_loader_runtime import _build_descriptor_ready_dataset


def test_build_evidence_package_and_report_runtime(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    manifests = build_training_manifests(
        descriptor_index_csv=root / "processed" / "manifests" / "descriptor_index.csv",
        data_root=root,
        categories=["capsule"],
    )

    train_dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        data_root=".",
        target_size=(16, 16),
        global_stats_path=manifests["global_stats_json"],
    )
    eval_dataset = DescriptorConditioningDataset(
        manifests["eval_csv"],
        data_root=".",
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
        diffusion_steps=50,
    )

    calibration_scores = []
    for sample in calibration_samples:
        scored = score_batch(
            model,
            sample.x_rgb.unsqueeze(0),
            sample.conditioning.descriptor_maps.unsqueeze(0),
            sample.conditioning.global_vector.unsqueeze(0),
            timestep=10,
        )
        calibration_scores.append(scored.anomaly_score.detach().cpu())
    calibration = fit_masi_calibration(torch.cat(calibration_scores), category="capsule", score_mode="noise_error")

    scored_eval = score_batch(
        model,
        eval_sample.x_rgb.unsqueeze(0),
        eval_sample.conditioning.descriptor_maps.unsqueeze(0),
        eval_sample.conditioning.global_vector.unsqueeze(0),
        timestep=10,
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
        source_paths={"rgb_path": eval_sample.rgb_path, "rgb_gt_path": eval_sample.rgb_gt_path},
    )
    report = render_templated_explanation(package)

    assert 0.0 <= package.severity_0_100 <= 100.0
    assert 0.0 <= package.confidence_0_100 <= 100.0
    assert package.top_regions
    assert abs(sum(package.evidence_breakdown.values()) - 100.0) < 1.0
    assert "Status:" in report
    assert "Observations:" in report


def test_evaluate_checkpoint_writes_evidence_outputs(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    train_result = train_model(
        data_root=root,
        descriptor_index_csv=root / "processed" / "manifests" / "descriptor_index.csv",
        category="capsule",
        epochs=1,
        batch_size=1,
        target_size=(16, 16),
        device="cpu",
        output_root=root / "runs",
        max_train_batches=1,
        max_eval_batches=1,
        max_visualizations=1,
    )
    best_checkpoint = Path(train_result["summary"]["best_checkpoint"])

    eval_result = evaluate_checkpoint(
        checkpoint_path=best_checkpoint,
        data_root=root,
        descriptor_index_csv=root / "processed" / "manifests" / "descriptor_index.csv",
        category="capsule",
        batch_size=1,
        target_size=(16, 16),
        device="cpu",
        output_root=root / "eval_runs",
        max_eval_batches=1,
        max_visualizations=1,
    )

    summary = eval_result["summary"]
    assert summary["evidence_packages_saved"] >= 1
    assert Path(summary["calibration_path"]).exists()
    assert Path(summary["evidence_index_path"]).exists()

    evidence_index = json.loads(Path(summary["evidence_index_path"]).read_text(encoding="utf-8"))
    assert len(evidence_index) >= 1
    assert Path(evidence_index[0]["package_path"]).exists()
    assert Path(evidence_index[0]["report_path"]).exists()
