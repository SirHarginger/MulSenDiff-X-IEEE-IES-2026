from pathlib import Path

import torch

from src.data_loader import DescriptorConditioningDataset, build_training_manifests
from src.evaluation.metrics import binary_f1_score, image_level_auroc, intersection_over_union
from src.inference.anomaly_scorer import score_batch
from src.models.mulsendiffx import MulSenDiffX
from src.training.train_diffusion import evaluate_checkpoint, train_model
from tests.test_data_loader_runtime import _build_descriptor_ready_dataset


def test_mulsendiffx_forward_and_anomaly_scoring_runtime(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    manifests = build_training_manifests(
        descriptor_index_csv=root / "processed" / "manifests" / "descriptor_index.csv",
        data_root=root,
        categories=["capsule"],
    )
    dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        data_root=".",
        target_size=(16, 16),
        global_stats_path=manifests["global_stats_json"],
    )

    samples = [dataset[0], dataset[1]]
    x_rgb = torch.stack([sample.x_rgb for sample in samples], dim=0)
    descriptor_maps = torch.stack([sample.conditioning.descriptor_maps for sample in samples], dim=0)
    global_vectors = torch.stack([sample.conditioning.global_vector for sample in samples], dim=0)
    timesteps = torch.tensor([10, 20], dtype=torch.long)

    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=dataset.descriptor_channels,
        global_dim=dataset.global_dim,
        base_channels=8,
        global_embedding_dim=16,
        time_embedding_dim=16,
        diffusion_steps=50,
    )
    predicted_noise = model(x_rgb, descriptor_maps, global_vectors, timesteps)
    assert predicted_noise.shape == x_rgb.shape

    outputs = model.training_outputs(x_rgb, descriptor_maps, global_vectors, timesteps=timesteps)
    assert outputs.loss.item() >= 0.0

    scored = score_batch(model, x_rgb, descriptor_maps, global_vectors, timestep=10)
    assert scored.anomaly_map.shape == (2, 1, 16, 16)
    assert scored.residual_map.shape == (2, 1, 16, 16)
    assert scored.anomaly_score.shape == (2,)


def test_basic_metrics_runtime() -> None:
    prediction = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert binary_f1_score(prediction, target) == 1.0
    assert intersection_over_union(prediction, target) == 1.0

    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert image_level_auroc(scores, labels) == 1.0


def test_train_model_and_evaluate_checkpoint_runtime(tmp_path: Path) -> None:
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
    assert best_checkpoint.exists()

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
    assert Path(eval_result["summary"]["run_dir"]).exists()
    assert eval_result["summary"]["eval_records"] >= 1
