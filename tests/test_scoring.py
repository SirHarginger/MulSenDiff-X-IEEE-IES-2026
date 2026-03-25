from pathlib import Path

import torch
import json

from src.data_loader import DescriptorConditioningDataset, build_training_manifests
from src.evaluation.metrics import (
    aupro,
    binary_f1_score,
    image_average_precision,
    image_level_auroc,
    intersection_over_union,
    pixel_level_auroc,
)
from src.inference.anomaly_scorer import score_batch
from src.models.mulsendiffx import MulSenDiffX
from src.training.train_diffusion import (
    evaluate_checkpoint,
    fit_global_descriptor_reference,
    resolve_score_weights,
    train_model,
)
from src.utils.visualization import save_evaluation_plot_bundle
from tests.test_data_loader_runtime import (
    _build_descriptor_ready_dataset,
    _build_multi_category_processed_manifest_ready_dataset,
)


def test_mulsendiffx_forward_and_anomaly_scoring_runtime(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    manifests = build_training_manifests(data_root=root, categories=["capsule"])
    dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        target_size=(16, 16),
        global_stats_path=manifests["global_stats_json"],
    )

    samples = [dataset[0], dataset[1]]
    x_rgb = torch.stack([sample.x_rgb for sample in samples], dim=0)
    descriptor_maps = torch.stack([sample.conditioning.descriptor_maps for sample in samples], dim=0)
    support_maps = torch.stack([sample.conditioning.support_maps for sample in samples], dim=0)
    global_vectors = torch.stack([sample.conditioning.global_vector for sample in samples], dim=0)
    timesteps = torch.tensor([10, 20], dtype=torch.long)

    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=dataset.descriptor_channels,
        global_dim=dataset.global_dim,
        base_channels=8,
        global_embedding_dim=16,
        time_embedding_dim=16,
        attention_heads=2,
        diffusion_steps=50,
    )
    latent = model.encode_rgb(x_rgb)
    predicted_noise = model(latent, descriptor_maps, global_vectors, timesteps)
    assert predicted_noise.shape == latent.shape

    outputs = model.training_outputs(x_rgb, descriptor_maps, global_vectors, timesteps=timesteps)
    assert outputs.loss.item() >= 0.0
    assert outputs.reconstructed_rgb.shape == x_rgb.shape
    assert outputs.diffusion_reconstruction_loss.item() >= 0.0
    assert outputs.edge_reconstruction_loss.item() >= 0.0

    calibration = fit_global_descriptor_reference(
        torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda batch: {
            "x_rgb": torch.stack([item.x_rgb for item in batch]),
            "descriptor_maps": torch.stack([item.conditioning.descriptor_maps for item in batch]),
            "support_maps": torch.stack([item.conditioning.support_maps for item in batch]),
            "global_vectors": torch.stack([item.conditioning.global_vector for item in batch]),
            "categories": [item.category for item in batch],
            "defect_labels": [item.defect_label for item in batch],
            "sample_ids": [item.sample_id for item in batch],
            "sample_names": [item.sample_name for item in batch],
            "is_anomalous": [item.is_anomalous for item in batch],
            "sample_dirs": [item.sample_dir for item in batch],
            "rgb_paths": [item.rgb_path for item in batch],
            "gt_mask_paths": [item.gt_mask_path for item in batch],
            "source_rgb_paths": [item.source_rgb_path for item in batch],
            "source_ir_paths": [item.source_ir_path for item in batch],
            "source_pointcloud_paths": [item.source_pointcloud_path for item in batch],
        }),
        category="capsule",
    )
    scored = score_batch(
        model,
        x_rgb,
        descriptor_maps,
        support_maps,
        global_vectors,
        timestep=10,
        multi_timestep_scoring_enabled=True,
        multi_timestep_fractions=[0.1, 0.2],
        anomaly_map_gaussian_sigma=1.5,
        lambda_residual=0.2,
        lambda_noise=0.3,
        lambda_descriptor_spatial=0.25,
        lambda_descriptor_global=0.8,
        global_descriptor_weight=0.1,
        global_descriptor_calibration=calibration,
    )
    assert scored.anomaly_map.shape == (2, 1, 16, 16)
    assert scored.residual_map.shape == (2, 1, 16, 16)
    assert scored.anomaly_score.shape == (2,)
    assert scored.global_descriptor_score.shape == (2,)
    assert scored.residual_component.shape == (2, 1, 16, 16)
    assert scored.noise_component.shape == (2, 1, 16, 16)
    assert scored.descriptor_spatial_component.shape == (2, 1, 16, 16)
    assert scored.timesteps_used == (5, 10)


def test_basic_metrics_runtime() -> None:
    prediction = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert binary_f1_score(prediction, target) == 1.0
    assert intersection_over_union(prediction, target) == 1.0

    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert image_level_auroc(scores, labels) == 1.0
    assert image_average_precision(scores, labels) == 1.0

    score_maps = [
        torch.tensor([[[0.0, 0.1], [0.8, 0.9]]], dtype=torch.float32),
        torch.tensor([[[0.0, 0.2], [0.1, 0.0]]], dtype=torch.float32),
    ]
    target_masks = [
        torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
    ]
    object_masks = [
        torch.ones((1, 2, 2), dtype=torch.float32),
        torch.ones((1, 2, 2), dtype=torch.float32),
    ]
    assert pixel_level_auroc(score_maps, target_masks, object_masks) == 1.0
    perfect_aupro = aupro(score_maps, target_masks, object_masks)
    assert 0.0 <= perfect_aupro <= 1.0
    assert perfect_aupro > 0.9


def test_resolve_score_weights_interpolates_between_warmup_and_main() -> None:
    schedule = {
        "warmup_fraction": 0.4,
        "transition_fraction": 0.6,
        "warmup": {
            "lambda_residual": 0.2,
            "lambda_noise": 0.3,
            "lambda_descriptor_spatial": 0.25,
            "lambda_descriptor_global": 0.8,
        },
        "main": {
            "lambda_residual": 0.6,
            "lambda_noise": 0.4,
            "lambda_descriptor_spatial": 0.25,
            "lambda_descriptor_global": 0.55,
        },
    }

    warmup = resolve_score_weights(
        total_epochs=50,
        epoch=20,
        lambda_residual=1.0,
        lambda_noise=0.5,
        lambda_descriptor_spatial=0.25,
        lambda_descriptor_global=0.3,
        score_weight_schedule=schedule,
    )
    assert warmup.lambda_residual == 0.2
    assert warmup.lambda_descriptor_global == 0.8

    blended = resolve_score_weights(
        total_epochs=50,
        epoch=30,
        lambda_residual=0.6,
        lambda_noise=0.4,
        lambda_descriptor_spatial=0.25,
        lambda_descriptor_global=0.55,
        score_weight_schedule=schedule,
    )
    assert 0.2 < blended.lambda_residual < 0.6
    assert 0.55 < blended.lambda_descriptor_global < 0.8

    main = resolve_score_weights(
        total_epochs=50,
        epoch=50,
        lambda_residual=0.6,
        lambda_noise=0.4,
        lambda_descriptor_spatial=0.25,
        lambda_descriptor_global=0.55,
        score_weight_schedule=schedule,
    )
    assert main.lambda_residual == 0.6
    assert main.lambda_descriptor_global == 0.55


def test_save_evaluation_plot_bundle_runtime(tmp_path: Path) -> None:
    image_score_data_path = tmp_path / "image_score_data.json"
    image_score_data_path.write_text(
        json.dumps(
            [
                {"category": "capsule", "defect_label": "good", "sample_id": "0", "sample_name": "good_0", "label": 0, "score": 0.1},
                {"category": "capsule", "defect_label": "good", "sample_id": "1", "sample_name": "good_1", "label": 0, "score": 0.2},
                {"category": "capsule", "defect_label": "crack", "sample_id": "2", "sample_name": "bad_0", "label": 1, "score": 0.8},
                {"category": "capsule", "defect_label": "crack", "sample_id": "3", "sample_name": "bad_1", "label": 1, "score": 0.9},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_paths = save_evaluation_plot_bundle(
        evaluation_metrics={
            "image_auroc": 1.0,
            "macro_image_auroc": 1.0,
            "image_auprc": 1.0,
            "macro_image_auprc": 1.0,
            "pixel_auroc": 0.9,
            "macro_pixel_auroc": 0.9,
            "aupro": 0.8,
            "macro_aupro": 0.8,
        },
        per_category={
            "capsule": {
                "image_auroc": 1.0,
                "image_auprc": 1.0,
                "pixel_auroc": 0.9,
                "aupro": 0.8,
            }
        },
        image_score_data_path=image_score_data_path,
        plot_dir=tmp_path / "plots",
    )
    assert set(plot_paths.keys()) == {
        "evaluation_summary",
        "per_category_core_metrics",
        "image_roc_curve",
        "image_pr_curve",
        "image_score_distribution",
    }
    for output_path in plot_paths.values():
        assert Path(output_path).exists()


def test_train_model_and_evaluate_checkpoint_runtime(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
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
        seed=123,
        log_every_n_steps=1,
    )
    best_checkpoint = Path(train_result["summary"]["best_checkpoint"])
    assert best_checkpoint.exists()
    assert (Path(train_result["summary"]["run_dir"]) / "logs" / "train_steps.jsonl").exists()
    train_config = json.loads((Path(train_result["summary"]["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert train_config["seed"] == 123
    assert train_config["resolved_device"] == "cpu"
    assert Path(train_config["localization_calibration_path"]).exists()
    assert "python_version" in train_config
    assert "torch_version" in train_config
    assert "git_commit" in train_config

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
        enable_llm_explanations=False,
        seed=321,
    )
    assert Path(eval_result["summary"]["run_dir"]).exists()
    assert eval_result["summary"]["eval_records"] >= 1
    assert eval_result["summary"]["seed"] == 321
    assert eval_result["summary"]["resolved_device"] == "cpu"
    assert Path(eval_result["summary"]["localization_calibration_path"]).exists()
    assert "image_auprc" in eval_result["summary"]
    assert "pixel_auroc" in eval_result["summary"]
    assert "aupro" in eval_result["summary"]
    image_score_data_path = Path(eval_result["summary"]["image_score_data_path"])
    assert image_score_data_path.exists()
    for plot_name in (
        "evaluation_summary.png",
        "per_category_core_metrics.png",
        "image_roc_curve.png",
        "image_pr_curve.png",
        "image_score_distribution.png",
    ):
        assert (Path(eval_result["summary"]["run_dir"]) / "plots" / plot_name).exists()


def test_cuda_first_runtime_falls_back_to_cpu(monkeypatch, tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)

    class _Probe:
        returncode = 1
        stdout = ""
        stderr = "driver unavailable"

    monkeypatch.setattr("src.training.train_diffusion.subprocess.run", lambda *args, **kwargs: _Probe())
    monkeypatch.setattr("src.training.train_diffusion.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("src.training.train_diffusion.torch.cuda.device_count", lambda: 0)

    result = train_model(
        data_root=root,
        category="capsule",
        epochs=1,
        batch_size=1,
        target_size=(16, 16),
        device_mode="cuda_first",
        output_root=root / "runs",
        max_train_batches=1,
        max_eval_batches=1,
        max_visualizations=1,
        base_channels=8,
        global_embedding_dim=16,
        time_embedding_dim=16,
        attention_heads=2,
        diffusion_steps=50,
        log_every_n_steps=1,
    )
    summary = result["summary"]
    assert summary["device"] == "cpu"
    assert summary["device_mode"] == "cuda_first"
    assert summary["amp_enabled"] is False
    assert "torch.cuda.is_available()" in summary["runtime_fallback_reason"]


def test_joint_train_and_eval_runtime_exports_macro_and_per_category_metrics(tmp_path: Path) -> None:
    root = _build_multi_category_processed_manifest_ready_dataset(tmp_path)
    train_result = train_model(
        data_root=root,
        categories=["capsule", "screw"],
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
        category_embedding_dim=8,
        attention_heads=2,
        diffusion_steps=50,
        log_every_n_steps=1,
    )
    summary = train_result["summary"]
    assert summary["training_mode"] == "shared_joint"
    assert summary["selected_categories"] == ["capsule", "screw"]
    best_checkpoint = Path(summary["best_checkpoint"])
    assert best_checkpoint.exists()

    eval_result = evaluate_checkpoint(
        checkpoint_path=best_checkpoint,
        data_root=root,
        categories=["capsule", "screw"],
        batch_size=1,
        target_size=(16, 16),
        device="cpu",
        output_root=root / "eval_runs",
        max_eval_batches=0,
        max_visualizations=1,
        enable_llm_explanations=False,
        seed=99,
    )
    eval_summary = eval_result["summary"]
    assert eval_summary["training_mode"] == "shared_joint"
    assert eval_summary["selected_categories"] == ["capsule", "screw"]
    assert Path(eval_summary["localization_calibration_path"]).exists()
    assert "macro_image_auroc" in eval_summary
    assert "macro_image_auprc" in eval_summary
    assert "macro_pixel_auroc" in eval_summary
    assert "macro_aupro" in eval_summary
    assert (Path(eval_summary["run_dir"]) / "metrics" / "per_category.json").exists()
    assert (Path(eval_summary["run_dir"]) / "metrics" / "per_category.csv").exists()
    preview_selection_path = Path(eval_summary["preview_selection_path"])
    assert preview_selection_path.exists()
    preview_selection = json.loads(preview_selection_path.read_text(encoding="utf-8"))
    assert preview_selection["strategy"] == "one_random_sample_per_category"
    assert {item["category"] for item in preview_selection["selected_samples"]} == {"capsule", "screw"}
    image_score_data_path = Path(eval_summary["image_score_data_path"])
    assert image_score_data_path.exists()
    for plot_name in (
        "evaluation_summary.png",
        "per_category_core_metrics.png",
        "image_roc_curve.png",
        "image_pr_curve.png",
        "image_score_distribution.png",
    ):
        assert (Path(eval_summary["run_dir"]) / "plots" / plot_name).exists()
