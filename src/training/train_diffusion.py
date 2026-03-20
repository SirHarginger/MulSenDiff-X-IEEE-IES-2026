from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data_loader import DescriptorConditioningDataset, build_training_manifests, collate_model_samples
from src.evaluation.metrics import binary_f1_score, image_level_auroc, intersection_over_union
from src.explainer.evidence_builder import build_evidence_package, save_evidence_package
from src.explainer.report import render_templated_explanation, save_templated_explanation
from src.inference.localization import estimate_object_mask, normalize_anomaly_map, threshold_anomaly_map
from src.inference.anomaly_scorer import score_batch
from src.inference.quantification import MasiCalibration, fit_masi_calibration
from src.models.mulsendiffx import MulSenDiffX
from src.utils.checkpoint import load_checkpoint
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import RunPaths, create_run_dir, write_history_csv, write_json
from src.utils.visualization import save_anomaly_panel, save_training_curves


@dataclass(frozen=True)
class DiffusionInputBatch:
    x_rgb: torch.Tensor
    M: torch.Tensor
    g: torch.Tensor
    categories: Sequence[str]
    defect_labels: Sequence[str]
    sample_ids: Sequence[str]
    is_anomalous: Sequence[bool]
    rgb_paths: Sequence[str]
    rgb_gt_paths: Sequence[str]
    ir_gt_paths: Sequence[str]
    pointcloud_gt_paths: Sequence[str]


def prepare_diffusion_batch(batch: Dict[str, object]) -> DiffusionInputBatch:
    return DiffusionInputBatch(
        x_rgb=batch["x_rgb"],  # type: ignore[arg-type]
        M=batch["descriptor_maps"],  # type: ignore[arg-type]
        g=batch["global_vectors"],  # type: ignore[arg-type]
        categories=batch["categories"],  # type: ignore[arg-type]
        defect_labels=batch["defect_labels"],  # type: ignore[arg-type]
        sample_ids=batch["sample_ids"],  # type: ignore[arg-type]
        is_anomalous=batch["is_anomalous"],  # type: ignore[arg-type]
        rgb_paths=batch["rgb_paths"],  # type: ignore[arg-type]
        rgb_gt_paths=batch["rgb_gt_paths"],  # type: ignore[arg-type]
        ir_gt_paths=batch["ir_gt_paths"],  # type: ignore[arg-type]
        pointcloud_gt_paths=batch["pointcloud_gt_paths"],  # type: ignore[arg-type]
    )


def build_diffusion_dataloaders(
    *,
    data_root: Path | str = "data",
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
) -> Dict[str, object]:
    data_root = Path(data_root)
    manifests = build_training_manifests(
        descriptor_index_csv=descriptor_index_csv,
        data_root=data_root,
        categories=[category],
    )

    train_dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
    )
    eval_dataset = DescriptorConditioningDataset(
        manifests["eval_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )

    return {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "manifests": manifests,
    }


def train_step(
    model: MulSenDiffX,
    batch: DiffusionInputBatch,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    clean_rgb = batch.x_rgb.to(device)
    descriptor_maps = batch.M.to(device)
    global_vector = batch.g.to(device)

    outputs = model.training_outputs(
        clean_rgb,
        descriptor_maps,
        global_vector,
        generator=generator,
    )
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": float(outputs.loss.detach().item()),
        "batch_shape": tuple(int(dim) for dim in clean_rgb.shape),
        "conditioning_dim": int(model.global_encoder.network[-1].out_features + model.time_encoder.embedding_dim),
        "timesteps_mean": float(outputs.timesteps.float().mean().item()),
        "timesteps_max": int(outputs.timesteps.max().item()),
        "sample_ids": list(batch.sample_ids),
    }


def run_training_epoch(
    model: MulSenDiffX,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_batches: int = 0,
    seed: int = 7,
) -> List[Dict[str, object]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    batch_reports: List[Dict[str, object]] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        metrics = train_step(model, batch, optimizer, device=device, generator=generator)
        batch_reports.append(
            {
                "batch_index": batch_index,
                "loss": round(float(metrics["loss"]), 6),
                "conditioning_dim": metrics["conditioning_dim"],
                "timesteps_mean": round(float(metrics["timesteps_mean"]), 4),
                "timesteps_max": metrics["timesteps_max"],
                "batch_shape": metrics["batch_shape"],
                "sample_ids": metrics["sample_ids"],
            }
        )
        if max_batches and len(batch_reports) >= max_batches:
            break
    return batch_reports


def run_eval_scoring(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    target_size: tuple[int, int],
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    max_batches: int = 0,
    visualization_dir: Path | None = None,
    evidence_dir: Path | None = None,
    calibration: MasiCalibration | None = None,
    max_visualizations: int = 6,
    seed: int = 17,
) -> Dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    reports: List[Dict[str, object]] = []
    evidence_index: List[Dict[str, object]] = []
    image_scores: List[torch.Tensor] = []
    image_labels: List[torch.Tensor] = []
    pixel_f1_values: List[float] = []
    pixel_iou_values: List[float] = []
    visualizations_saved = 0

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.g.to(device),
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        image_scores.append(scored.anomaly_score.detach().cpu())
        labels = torch.tensor([1 if value else 0 for value in batch.is_anomalous], dtype=torch.float32)
        image_labels.append(labels)

        normalized_maps = normalize_anomaly_map(scored.anomaly_map.detach().cpu())
        object_masks = estimate_object_mask(batch.x_rgb.detach().cpu(), threshold=object_mask_threshold)
        predicted_masks = threshold_anomaly_map(
            normalized_maps,
            percentile=pixel_threshold_percentile,
            object_mask=object_masks,
            normalized=True,
        )

        for sample_offset, sample_id in enumerate(batch.sample_ids):
            gt_mask = load_gt_mask(batch.rgb_gt_paths[sample_offset], target_size=target_size)
            if gt_mask is not None:
                pred_mask = predicted_masks[sample_offset]
                pixel_f1_values.append(binary_f1_score(pred_mask, gt_mask))
                pixel_iou_values.append(intersection_over_union(pred_mask, gt_mask))

            panel_path: Path | None = None
            if visualization_dir is not None and visualizations_saved < max_visualizations:
                panel_path = visualization_dir / f"{batch_index:03d}_{sample_id}_{batch.defect_labels[sample_offset]}.png"
                save_anomaly_panel(
                    rgb=batch.x_rgb[sample_offset].detach().cpu(),
                    reconstructed_rgb=scored.reconstructed_rgb[sample_offset].detach().cpu(),
                    anomaly_map=normalized_maps[sample_offset].detach().cpu(),
                    score_map=scored.score_basis_map[sample_offset].detach().cpu(),
                    score_map_label=scored.score_basis_label,
                    support_map=scored.descriptor_support[sample_offset].detach().cpu(),
                    gt_mask=gt_mask,
                    anomaly_score=float(scored.anomaly_score[sample_offset].item()),
                    title=f"{batch.categories[sample_offset]} / {batch.defect_labels[sample_offset]} / {sample_id}",
                    path=panel_path,
                )
                visualizations_saved += 1

            if evidence_dir is not None and calibration is not None:
                package = build_evidence_package(
                    category=batch.categories[sample_offset],
                    split="test",
                    defect_label=batch.defect_labels[sample_offset],
                    sample_id=sample_id,
                    is_anomalous=bool(batch.is_anomalous[sample_offset]),
                    score_mode=score_mode,
                    score_label=scored.score_basis_label,
                    raw_score=float(scored.anomaly_score[sample_offset].item()),
                    calibration=calibration,
                    normalized_anomaly_map=normalized_maps[sample_offset].detach().cpu(),
                    score_basis_map=scored.score_basis_map[sample_offset].detach().cpu(),
                    descriptor_support=scored.descriptor_support[sample_offset].detach().cpu(),
                    object_mask=object_masks[sample_offset].detach().cpu(),
                    predicted_mask=predicted_masks[sample_offset].detach().cpu(),
                    descriptor_maps=batch.M[sample_offset].detach().cpu(),
                    source_paths={
                        "rgb_path": batch.rgb_paths[sample_offset],
                        "rgb_gt_path": batch.rgb_gt_paths[sample_offset],
                        "ir_gt_path": batch.ir_gt_paths[sample_offset],
                        "pointcloud_gt_path": batch.pointcloud_gt_paths[sample_offset],
                        "visualization_path": str(panel_path) if panel_path is not None else "",
                    },
                )
                package_path = evidence_dir / "packages" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.json"
                report_path = evidence_dir / "reports" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.md"
                save_evidence_package(package, package_path)
                save_templated_explanation(render_templated_explanation(package), report_path)
                evidence_index.append(
                    {
                        "category": batch.categories[sample_offset],
                        "defect_label": batch.defect_labels[sample_offset],
                        "sample_id": sample_id,
                        "is_anomalous": bool(batch.is_anomalous[sample_offset]),
                        "raw_score": round(float(scored.anomaly_score[sample_offset].item()), 6),
                        "severity_0_100": round(float(package.severity_0_100), 3),
                        "confidence_0_100": round(float(package.confidence_0_100), 3),
                        "status": package.status,
                        "package_path": str(package_path),
                        "report_path": str(report_path),
                    }
                )

        reports.append(
            {
                "batch_index": batch_index,
                "mean_anomaly_score": round(float(scored.anomaly_score.mean().item()), 6),
                "max_anomaly_score": round(float(scored.anomaly_score.max().item()), 6),
                "score_mode": score_mode,
                "mean_score_basis": round(float(scored.score_basis_map.mean().item()), 6),
                "mean_residual": round(float(scored.residual_map.mean().item()), 6),
                "mean_noise_error": round(float(scored.noise_error_map.mean().item()), 6),
                "mean_descriptor_support": round(float(scored.descriptor_support.mean().item()), 6),
                "sample_ids": list(batch.sample_ids),
            }
        )
        if max_batches and len(reports) >= max_batches:
            break

    image_scores_tensor = torch.cat(image_scores) if image_scores else torch.empty(0)
    image_labels_tensor = torch.cat(image_labels) if image_labels else torch.empty(0)
    good_scores = image_scores_tensor[image_labels_tensor == 0]
    anomaly_scores = image_scores_tensor[image_labels_tensor == 1]

    image_auroc = image_level_auroc(image_scores_tensor, image_labels_tensor) if image_scores else 0.0
    threshold = 0.0
    image_f1 = 0.0
    if len(good_scores) and len(anomaly_scores):
        threshold = float((good_scores.mean() + anomaly_scores.mean()).item() / 2.0)
        image_f1 = binary_f1_score(image_scores_tensor, image_labels_tensor, threshold=threshold)

    evidence_index_path = ""
    if evidence_dir is not None:
        evidence_index_path = str(write_json(evidence_dir / "index.json", evidence_index))

    return {
        "batch_reports": reports,
        "image_auroc": round(float(image_auroc), 6),
        "image_f1": round(float(image_f1), 6),
        "image_threshold": round(float(threshold), 6),
        "mean_good_score": round(float(good_scores.mean().item()), 6) if len(good_scores) else 0.0,
        "mean_anomalous_score": round(float(anomaly_scores.mean().item()), 6) if len(anomaly_scores) else 0.0,
        "pixel_f1": round(sum(pixel_f1_values) / max(len(pixel_f1_values), 1), 6),
        "pixel_iou": round(sum(pixel_iou_values) / max(len(pixel_iou_values), 1), 6),
        "pixel_samples": len(pixel_f1_values),
        "visualizations_saved": visualizations_saved,
        "evidence_packages_saved": len(evidence_index),
        "evidence_index_path": evidence_index_path,
    }


def run_smoke_experiment(
    *,
    data_root: Path | str = "data",
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    batch_size: int = 2,
    max_batches: int = 3,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    learning_rate: float = 1e-4,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
) -> Dict[str, object]:
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
    )

    train_dataset: DescriptorConditioningDataset = loaders["train_dataset"]  # type: ignore[assignment]
    if len(train_dataset) == 0:
        raise ValueError(f"No training records available for smoke experiment category={category!r}")

    torch_device = torch.device(device)
    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=train_dataset.descriptor_channels,
        global_dim=train_dataset.global_dim,
    ).to(torch_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    batch_reports = run_training_epoch(
        model,
        loaders["train_loader"],  # type: ignore[arg-type]
        optimizer,
        device=torch_device,
        max_batches=max_batches,
    )
    eval_reports = run_eval_scoring(
        model,
        loaders["eval_loader"],  # type: ignore[arg-type]
        device=torch_device,
        target_size=target_size,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        pixel_threshold_percentile=pixel_threshold_percentile,
        object_mask_threshold=object_mask_threshold,
        max_batches=1,
    )

    manifests = loaders["manifests"]  # type: ignore[assignment]
    summary = {
        "category": category,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "target_size": target_size,
        "device": str(torch_device),
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "anomaly_timestep": anomaly_timestep,
        "descriptor_weight": descriptor_weight,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "object_mask_threshold": object_mask_threshold,
        "train_manifest_csv": str(manifests["train_csv"]),
        "eval_manifest_csv": str(manifests["eval_csv"]),
        "global_stats_json": str(manifests["global_stats_json"]),
        "train_records": len(train_dataset),
        "eval_records": len(loaders["eval_dataset"]),  # type: ignore[arg-type]
        "batches_run": len(batch_reports),
        "batch_reports": batch_reports,
        "eval_reports": eval_reports["batch_reports"],
        "image_auroc": eval_reports["image_auroc"],
        "pixel_iou": eval_reports["pixel_iou"],
        "mean_loss": round(sum(report["loss"] for report in batch_reports) / max(len(batch_reports), 1), 6),
    }

    out_path = Path(data_root) / "processed" / "reports" / f"smoke_experiment_{category}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "summary_path": out_path}


def train_model(
    *,
    data_root: Path | str = "data",
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    epochs: int = 5,
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    learning_rate: float = 1e-4,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    max_visualizations: int = 6,
) -> Dict[str, Any]:
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
    )

    train_dataset: DescriptorConditioningDataset = loaders["train_dataset"]  # type: ignore[assignment]
    eval_dataset: DescriptorConditioningDataset = loaders["eval_dataset"]  # type: ignore[assignment]
    torch_device = torch.device(device)
    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=train_dataset.descriptor_channels,
        global_dim=train_dataset.global_dim,
    ).to(torch_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    run_paths = create_run_dir(output_root=output_root, run_name=run_name, category=category)
    manifests = loaders["manifests"]  # type: ignore[assignment]
    config_payload = {
        "category": category,
        "epochs": epochs,
        "batch_size": batch_size,
        "target_size": target_size,
        "device": str(torch_device),
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "anomaly_timestep": anomaly_timestep,
        "descriptor_weight": descriptor_weight,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "object_mask_threshold": object_mask_threshold,
        "descriptor_index_csv": str(descriptor_index_csv),
        "train_manifest_csv": str(manifests["train_csv"]),
        "eval_manifest_csv": str(manifests["eval_csv"]),
        "global_stats_json": str(manifests["global_stats_json"]),
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
    }
    write_json(run_paths.root / "config.json", config_payload)

    history: List[Dict[str, Any]] = []
    best_auroc = float("-inf")
    best_checkpoint_path: Path | None = None

    for epoch in range(1, epochs + 1):
        train_reports = run_training_epoch(
            model,
            loaders["train_loader"],  # type: ignore[arg-type]
            optimizer,
            device=torch_device,
            max_batches=max_train_batches,
            seed=7 + epoch,
        )
        eval_summary = run_eval_scoring(
            model,
            loaders["eval_loader"],  # type: ignore[arg-type]
            device=torch_device,
            target_size=target_size,
            score_mode=score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            pixel_threshold_percentile=pixel_threshold_percentile,
            object_mask_threshold=object_mask_threshold,
            max_batches=max_eval_batches,
            visualization_dir=run_paths.predictions / f"epoch_{epoch:03d}",
            max_visualizations=max_visualizations,
            seed=17 + epoch,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(sum(report["loss"] for report in train_reports) / max(len(train_reports), 1), 6),
            "image_auroc": eval_summary["image_auroc"],
            "image_f1": eval_summary["image_f1"],
            "pixel_f1": eval_summary["pixel_f1"],
            "pixel_iou": eval_summary["pixel_iou"],
            "mean_good_score": eval_summary["mean_good_score"],
            "mean_anomalous_score": eval_summary["mean_anomalous_score"],
        }
        history.append(epoch_summary)

        write_json(run_paths.metrics / f"epoch_{epoch:03d}.json", {
            "epoch": epoch,
            "train_reports": train_reports,
            "eval_summary": eval_summary,
        })

        save_checkpoint(
            run_paths.checkpoints / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_summary,
            config=config_payload,
        )
        if eval_summary["image_auroc"] >= best_auroc:
            best_auroc = float(eval_summary["image_auroc"])
            best_checkpoint_path = save_checkpoint(
                run_paths.checkpoints / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_summary,
                config=config_payload,
            )

        write_history_csv(run_paths.logs / "history.csv", history)
        write_json(run_paths.logs / "history.json", history)
        save_training_curves(history, run_paths.plots / "training_curves.png")

    final_summary = {
        "run_dir": str(run_paths.root),
        "category": category,
        "epochs": epochs,
        "device": str(torch_device),
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "train_records": len(train_dataset),
        "eval_records": len(eval_dataset),
        "best_image_auroc": round(best_auroc, 6) if best_auroc != float("-inf") else 0.0,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else "",
        "history_rows": len(history),
    }
    write_json(run_paths.root / "summary.json", final_summary)
    return {
        "run_paths": run_paths,
        "history": history,
        "summary": final_summary,
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: Path | str,
    data_root: Path | str = "data",
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    max_eval_batches: int = 0,
    max_visualizations: int = 10,
) -> Dict[str, Any]:
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
    )
    eval_dataset: DescriptorConditioningDataset = loaders["eval_dataset"]  # type: ignore[assignment]
    torch_device = torch.device(device)
    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=eval_dataset.descriptor_channels,
        global_dim=eval_dataset.global_dim,
    ).to(torch_device)
    payload = load_checkpoint(checkpoint_path, model=model, map_location=torch_device)

    run_paths = create_run_dir(output_root=output_root, run_name=run_name or "eval", category=category)
    evidence_root = run_paths.root / "evidence"
    calibration_scores = collect_reference_scores(
        model,
        loaders["train_loader"],  # type: ignore[arg-type]
        device=torch_device,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        object_mask_threshold=object_mask_threshold,
    )
    calibration = fit_masi_calibration(calibration_scores, category=category, score_mode=score_mode)
    calibration_path = write_json(evidence_root / "calibration.json", calibration.to_dict())
    eval_summary = run_eval_scoring(
        model,
        loaders["eval_loader"],  # type: ignore[arg-type]
        device=torch_device,
        target_size=target_size,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        pixel_threshold_percentile=pixel_threshold_percentile,
        object_mask_threshold=object_mask_threshold,
        max_batches=max_eval_batches,
        visualization_dir=run_paths.predictions / "eval",
        evidence_dir=evidence_root,
        calibration=calibration,
        max_visualizations=max_visualizations,
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "loaded_epoch": int(payload.get("epoch", 0)),
        "category": category,
        "device": str(torch_device),
        "score_mode": score_mode,
        "eval_records": len(eval_dataset),
        "image_auroc": eval_summary["image_auroc"],
        "image_f1": eval_summary["image_f1"],
        "pixel_f1": eval_summary["pixel_f1"],
        "pixel_iou": eval_summary["pixel_iou"],
        "mean_good_score": eval_summary["mean_good_score"],
        "mean_anomalous_score": eval_summary["mean_anomalous_score"],
        "visualizations_saved": eval_summary["visualizations_saved"],
        "evidence_packages_saved": eval_summary["evidence_packages_saved"],
        "calibration_path": str(calibration_path),
        "evidence_index_path": eval_summary["evidence_index_path"],
        "run_dir": str(run_paths.root),
    }
    write_json(run_paths.metrics / "evaluation.json", eval_summary)
    write_json(run_paths.root / "summary.json", summary)
    return {"run_paths": run_paths, "summary": summary, "eval_summary": eval_summary}


def collect_reference_scores(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    object_mask_threshold: float,
    max_batches: int = 0,
    seed: int = 29,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    collected: List[torch.Tensor] = []

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.g.to(device),
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        collected.append(scored.anomaly_score.detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break

    return torch.cat(collected) if collected else torch.empty(0)


def load_gt_mask(path_like: str, *, target_size: tuple[int, int]) -> torch.Tensor | None:
    path = Path(path_like) if path_like else None
    if not path or not path.exists():
        return None
    image = Image.open(path).convert("L")
    image = TF.resize(image, list(target_size), interpolation=InterpolationMode.NEAREST)
    tensor = TF.pil_to_tensor(image).float() / 255.0
    return (tensor > 0.5).float()
