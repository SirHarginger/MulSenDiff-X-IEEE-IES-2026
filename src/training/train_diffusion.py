from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data_loader import (
    GLOBAL_FEATURE_NAMES,
    DescriptorConditioningDataset,
    RGBNormalizationStats,
    build_training_manifests,
    collate_model_samples,
    compute_masked_rgb_normalization_stats,
    save_rgb_normalization_stats,
)
from src.evaluation.metrics import binary_f1_score, image_level_auroc, intersection_over_union
from src.explainer.evidence_builder import build_evidence_package, save_evidence_package
from src.explainer.report import render_templated_explanation, save_templated_explanation
from src.explainer.retriever import retrieve_context_for_evidence
from src.inference.anomaly_scorer import score_batch
from src.inference.global_descriptor_scoring import (
    GlobalDescriptorCalibration,
    fit_global_descriptor_calibration,
)
from src.inference.localization import estimate_object_mask, normalize_anomaly_map, threshold_anomaly_map
from src.inference.quantification import MasiCalibration, fit_masi_calibration
from src.models.llm_explainer import (
    build_llm_explanation,
    build_prompt_payload,
    render_llm_explanation_markdown,
    save_explanation_bundle,
)
from src.models.mulsendiffx import MulSenDiffX
from src.utils.checkpoint import load_checkpoint, read_checkpoint_payload, save_checkpoint
from src.utils.logger import RunPaths, append_jsonl, create_run_dir, write_history_csv, write_json
from src.utils.visualization import save_anomaly_panel, save_training_curves


@dataclass(frozen=True)
class DiffusionInputBatch:
    x_rgb: torch.Tensor
    M: torch.Tensor
    S: torch.Tensor
    g: torch.Tensor
    categories: Sequence[str]
    defect_labels: Sequence[str]
    sample_ids: Sequence[str]
    sample_names: Sequence[str]
    is_anomalous: Sequence[bool]
    sample_dirs: Sequence[str]
    rgb_paths: Sequence[str]
    gt_mask_paths: Sequence[str]
    source_rgb_paths: Sequence[str]
    source_ir_paths: Sequence[str]
    source_pointcloud_paths: Sequence[str]
    crop_boxes_xyxy: Sequence[tuple[int, int, int, int] | None]
    crop_source_sizes_hw: Sequence[tuple[int, int]]
    rgb_normalization_modes: Sequence[str]


@dataclass(frozen=True)
class RuntimeSettings:
    requested_mode: str
    resolved_device: torch.device
    use_amp: bool
    fallback_reason: str = ""


@dataclass(frozen=True)
class ScoreWeights:
    lambda_residual: float
    lambda_noise: float
    lambda_descriptor_spatial: float
    lambda_descriptor_global: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "lambda_residual": round(float(self.lambda_residual), 6),
            "lambda_noise": round(float(self.lambda_noise), 6),
            "lambda_descriptor_spatial": round(float(self.lambda_descriptor_spatial), 6),
            "lambda_descriptor_global": round(float(self.lambda_descriptor_global), 6),
        }


def prepare_diffusion_batch(batch: Dict[str, object]) -> DiffusionInputBatch:
    return DiffusionInputBatch(
        x_rgb=batch["x_rgb"],  # type: ignore[arg-type]
        M=batch["descriptor_maps"],  # type: ignore[arg-type]
        S=batch["support_maps"],  # type: ignore[arg-type]
        g=batch["global_vectors"],  # type: ignore[arg-type]
        categories=batch["categories"],  # type: ignore[arg-type]
        defect_labels=batch["defect_labels"],  # type: ignore[arg-type]
        sample_ids=batch["sample_ids"],  # type: ignore[arg-type]
        sample_names=batch["sample_names"],  # type: ignore[arg-type]
        is_anomalous=batch["is_anomalous"],  # type: ignore[arg-type]
        sample_dirs=batch["sample_dirs"],  # type: ignore[arg-type]
        rgb_paths=batch["rgb_paths"],  # type: ignore[arg-type]
        gt_mask_paths=batch["gt_mask_paths"],  # type: ignore[arg-type]
        source_rgb_paths=batch["source_rgb_paths"],  # type: ignore[arg-type]
        source_ir_paths=batch["source_ir_paths"],  # type: ignore[arg-type]
        source_pointcloud_paths=batch["source_pointcloud_paths"],  # type: ignore[arg-type]
        crop_boxes_xyxy=batch.get("crop_boxes_xyxy", [None] * len(batch["sample_ids"])),  # type: ignore[arg-type]
        crop_source_sizes_hw=batch.get(
            "crop_source_sizes_hw",
            [(int(batch["x_rgb"].shape[-2]), int(batch["x_rgb"].shape[-1]))] * len(batch["sample_ids"]),  # type: ignore[index]
        ),  # type: ignore[arg-type]
        rgb_normalization_modes=batch.get("rgb_normalization_modes", ["none"] * len(batch["sample_ids"])),  # type: ignore[arg-type]
    )


def _coerce_rgb_normalization_stats(
    value: RGBNormalizationStats | Dict[str, object] | None,
) -> RGBNormalizationStats | None:
    if value is None:
        return None
    if isinstance(value, RGBNormalizationStats):
        return value
    return RGBNormalizationStats(
        category=str(value.get("category", "")),
        mode=str(value.get("mode", "none")),
        mean=tuple(float(component) for component in value.get("mean", [0.0, 0.0, 0.0])),
        std=tuple(max(float(component), 1e-6) for component in value.get("std", [1.0, 1.0, 1.0])),
        masked_pixels=int(value.get("masked_pixels", 0)),
    )


def _resolve_requested_mode(device: str, device_mode: str) -> str:
    requested = (device_mode or device or "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "cuda_first"}:
        raise ValueError(f"Unsupported device mode {requested!r}. Expected 'cpu', 'cuda', or 'cuda_first'.")
    return requested


def _autocast_context(device: torch.device, *, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _validate_model_runtime(
    model: MulSenDiffX,
    batch: DiffusionInputBatch,
    *,
    device: torch.device,
    use_amp: bool,
    run_backward: bool,
) -> None:
    clean_rgb = batch.x_rgb.to(device)
    descriptor_maps = batch.M.to(device)
    global_vector = batch.g.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    model.zero_grad(set_to_none=True)
    if run_backward:
        model.train()
        with _autocast_context(device, use_amp=use_amp):
            outputs = model.training_outputs(
                clean_rgb,
                descriptor_maps,
                global_vector,
                generator=generator,
            )
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)
    else:
        model.eval()
        with torch.no_grad():
            with _autocast_context(device, use_amp=use_amp):
                model.training_outputs(
                    clean_rgb,
                    descriptor_maps,
                    global_vector,
                    generator=generator,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_runtime(
    *,
    requested_mode: str,
    model: MulSenDiffX,
    validation_batch: DiffusionInputBatch,
    run_backward: bool,
) -> RuntimeSettings:
    if requested_mode == "cpu":
        model.to(torch.device("cpu"))
        return RuntimeSettings(requested_mode=requested_mode, resolved_device=torch.device("cpu"), use_amp=False)

    static_failures: List[str] = []
    try:
        probe = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            stderr = probe.stderr.strip() or probe.stdout.strip() or "unknown nvidia-smi failure"
            static_failures.append(f"nvidia-smi failed: {stderr}")
    except FileNotFoundError:
        static_failures.append("nvidia-smi not found")

    if not torch.cuda.is_available():
        static_failures.append("torch.cuda.is_available() returned False")
    if torch.cuda.device_count() <= 0:
        static_failures.append("torch.cuda.device_count() returned 0")

    if static_failures:
        reason = "; ".join(static_failures)
        if requested_mode == "cuda":
            raise RuntimeError(reason)
        model.to(torch.device("cpu"))
        return RuntimeSettings(
            requested_mode=requested_mode,
            resolved_device=torch.device("cpu"),
            use_amp=False,
            fallback_reason=reason,
        )

    cuda_device = torch.device("cuda")
    model.to(cuda_device)
    try:
        _validate_model_runtime(
            model,
            validation_batch,
            device=cuda_device,
            use_amp=True,
            run_backward=run_backward,
        )
    except Exception as exc:
        reason = f"cuda validation failed: {exc}"
        if requested_mode == "cuda":
            raise RuntimeError(reason) from exc
        model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return RuntimeSettings(
            requested_mode=requested_mode,
            resolved_device=torch.device("cpu"),
            use_amp=False,
            fallback_reason=reason,
        )
    return RuntimeSettings(requested_mode=requested_mode, resolved_device=cuda_device, use_amp=True)


def resolve_score_weights(
    *,
    total_epochs: int,
    epoch: int | None,
    lambda_residual: float,
    lambda_noise: float,
    lambda_descriptor_spatial: float,
    lambda_descriptor_global: float,
    score_weight_schedule: Dict[str, object] | None,
) -> ScoreWeights:
    if epoch is not None and total_epochs > 0 and score_weight_schedule:
        warmup_fraction = float(score_weight_schedule.get("warmup_fraction", 0.4))
        phase_key = "warmup" if (epoch / max(total_epochs, 1)) <= warmup_fraction else "main"
        phase_payload = score_weight_schedule.get(phase_key)
        if isinstance(phase_payload, dict):
            return ScoreWeights(
                lambda_residual=float(phase_payload.get("lambda_residual", lambda_residual)),
                lambda_noise=float(phase_payload.get("lambda_noise", lambda_noise)),
                lambda_descriptor_spatial=float(
                    phase_payload.get("lambda_descriptor_spatial", lambda_descriptor_spatial)
                ),
                lambda_descriptor_global=float(
                    phase_payload.get("lambda_descriptor_global", lambda_descriptor_global)
                ),
            )
    return ScoreWeights(
        lambda_residual=float(lambda_residual),
        lambda_noise=float(lambda_noise),
        lambda_descriptor_spatial=float(lambda_descriptor_spatial),
        lambda_descriptor_global=float(lambda_descriptor_global),
    )


def _denormalize_rgb_tensor(
    rgb: torch.Tensor,
    stats: RGBNormalizationStats | None,
) -> torch.Tensor:
    if stats is None or stats.mode == "none":
        return rgb
    mean = torch.tensor(stats.mean, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(stats.std, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return rgb * std + mean


def build_diffusion_dataloaders(
    *,
    data_root: Path | str = "data",
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    object_crop_enabled: bool = False,
    object_crop_margin_ratio: float = 0.10,
    object_crop_mask_source: str = "rgb_then_pc_density",
    rgb_normalization_mode: str = "none",
    rgb_normalization_stats: RGBNormalizationStats | Dict[str, object] | None = None,
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
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
    )
    eval_dataset = DescriptorConditioningDataset(
        manifests["eval_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
    )

    resolved_rgb_normalization_stats = _coerce_rgb_normalization_stats(rgb_normalization_stats)
    if rgb_normalization_mode != "none":
        if resolved_rgb_normalization_stats is None:
            resolved_rgb_normalization_stats = compute_masked_rgb_normalization_stats(train_dataset, category=category)
        train_dataset.set_rgb_normalization_stats(resolved_rgb_normalization_stats)
        eval_dataset.set_rgb_normalization_stats(resolved_rgb_normalization_stats)

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
        "rgb_normalization_stats": resolved_rgb_normalization_stats,
    }


def build_model(
    *,
    descriptor_channels: int,
    global_dim: int,
    base_channels: int = 32,
    global_embedding_dim: int = 128,
    time_embedding_dim: int = 128,
    attention_heads: int = 4,
    latent_channels: int = 4,
    diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    noise_schedule: str = "cosine",
    autoencoder_reconstruction_weight: float = 0.1,
) -> MulSenDiffX:
    return MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=descriptor_channels,
        global_dim=global_dim,
        base_channels=base_channels,
        global_embedding_dim=global_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        attention_heads=attention_heads,
        latent_channels=latent_channels,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_schedule=noise_schedule,
        autoencoder_reconstruction_weight=autoencoder_reconstruction_weight,
    )


def train_step(
    model: MulSenDiffX,
    batch: DiffusionInputBatch,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    generator: torch.Generator | None = None,
) -> Dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    clean_rgb = batch.x_rgb.to(device)
    descriptor_maps = batch.M.to(device)
    global_vector = batch.g.to(device)

    with _autocast_context(device, use_amp=use_amp):
        outputs = model.training_outputs(
            clean_rgb,
            descriptor_maps,
            global_vector,
            generator=generator,
        )
    if use_amp and scaler is not None:
        scaler.scale(outputs.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "loss": float(outputs.loss.detach().item()),
        "noise_loss": float(outputs.noise_loss.detach().item()),
        "autoencoder_reconstruction_loss": float(outputs.autoencoder_reconstruction_loss.detach().item()),
        "batch_shape": tuple(int(dim) for dim in clean_rgb.shape),
        "latent_shape": tuple(int(dim) for dim in outputs.clean_latent.shape),
        "conditioning_dim": int(model.global_embedding_dim + model.time_embedding_dim),
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
    use_amp: bool = False,
    epoch: int,
    max_batches: int = 0,
    seed: int = 7,
    log_every_n_steps: int = 10,
    log_to_jsonl: bool = True,
    step_log_path: Path | None = None,
) -> List[Dict[str, object]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_reports: List[Dict[str, object]] = []
    running_loss = 0.0
    start_time = perf_counter()
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        metrics = train_step(
            model,
            batch,
            optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            generator=generator,
        )
        running_loss += float(metrics["loss"])
        report = {
            "epoch": epoch,
            "batch_index": batch_index,
            "loss": round(float(metrics["loss"]), 6),
            "noise_loss": round(float(metrics["noise_loss"]), 6),
            "autoencoder_reconstruction_loss": round(float(metrics["autoencoder_reconstruction_loss"]), 6),
            "conditioning_dim": metrics["conditioning_dim"],
            "timesteps_mean": round(float(metrics["timesteps_mean"]), 4),
            "timesteps_max": metrics["timesteps_max"],
            "batch_shape": metrics["batch_shape"],
            "latent_shape": metrics["latent_shape"],
            "sample_ids": metrics["sample_ids"],
        }
        batch_reports.append(report)

        steps_done = batch_index + 1
        if total_batches > 0 and (
            steps_done == total_batches
            or (log_every_n_steps > 0 and steps_done % log_every_n_steps == 0)
        ):
            elapsed = perf_counter() - start_time
            steps_per_second = steps_done / max(elapsed, 1e-6)
            remaining_steps = max(total_batches - steps_done, 0)
            eta_seconds = remaining_steps / max(steps_per_second, 1e-6)
            log_payload = {
                "event": "train_step",
                "epoch": epoch,
                "step": steps_done,
                "total_steps": total_batches,
                "loss": report["loss"],
                "noise_loss": report["noise_loss"],
                "autoencoder_reconstruction_loss": report["autoencoder_reconstruction_loss"],
                "running_loss": round(running_loss / steps_done, 6),
                "steps_per_second": round(steps_per_second, 4),
                "eta_seconds": round(float(eta_seconds), 2),
            }
            print(
                f"[train] epoch={epoch} step={steps_done}/{total_batches} "
                f"loss={report['loss']:.6f} noise={report['noise_loss']:.6f} "
                f"ae={report['autoencoder_reconstruction_loss']:.6f} "
                f"avg={log_payload['running_loss']:.6f} eta={log_payload['eta_seconds']:.1f}s"
            )
            if log_to_jsonl and step_log_path is not None:
                append_jsonl(step_log_path, log_payload)

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
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | None = None,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    max_batches: int = 0,
    visualization_dir: Path | None = None,
    evidence_dir: Path | None = None,
    calibration: MasiCalibration | None = None,
    max_visualizations: int = 6,
    knowledge_base_root: Path | None = None,
    retrieval_top_k: int = 3,
    enable_llm_explanations: bool = True,
    rgb_normalization_stats: RGBNormalizationStats | None = None,
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
    llm_explanations_saved = 0

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        image_scores.append(scored.anomaly_score.detach().cpu())
        labels = torch.tensor([1 if value else 0 for value in batch.is_anomalous], dtype=torch.float32)
        image_labels.append(labels)

        normalized_maps = normalize_anomaly_map(scored.anomaly_map.detach().cpu())
        object_masks = scored.object_mask.detach().cpu()
        predicted_masks = threshold_anomaly_map(
            normalized_maps,
            percentile=pixel_threshold_percentile,
            object_mask=object_masks,
            normalized=True,
        )

        for sample_offset, sample_id in enumerate(batch.sample_ids):
            gt_mask = load_gt_mask(
                batch.gt_mask_paths[sample_offset],
                target_size=target_size,
                crop_box_xyxy=batch.crop_boxes_xyxy[sample_offset],
                crop_source_size_hw=batch.crop_source_sizes_hw[sample_offset],
            )
            if gt_mask is not None:
                pred_mask = predicted_masks[sample_offset]
                pixel_f1_values.append(binary_f1_score(pred_mask, gt_mask))
                pixel_iou_values.append(intersection_over_union(pred_mask, gt_mask))

            panel_path: Path | None = None
            if visualization_dir is not None and visualizations_saved < max_visualizations:
                panel_path = visualization_dir / f"{batch_index:03d}_{sample_id}_{batch.defect_labels[sample_offset]}.png"
                save_anomaly_panel(
                    rgb=_denormalize_rgb_tensor(
                        batch.x_rgb[sample_offset].detach().cpu(),
                        rgb_normalization_stats,
                    ).clamp(0.0, 1.0),
                    reconstructed_rgb=_denormalize_rgb_tensor(
                        scored.reconstructed_rgb[sample_offset].detach().cpu(),
                        rgb_normalization_stats,
                    ).clamp(0.0, 1.0),
                    anomaly_map=normalized_maps[sample_offset].detach().cpu(),
                    score_map=scored.score_basis_map[sample_offset].detach().cpu(),
                    score_map_label=scored.score_basis_label,
                    support_map=scored.descriptor_support[sample_offset].detach().cpu(),
                    gt_mask=gt_mask,
                    anomaly_score=float(scored.anomaly_score[sample_offset].item()),
                    title=f"{batch.categories[sample_offset]} / {batch.defect_labels[sample_offset]} / {batch.sample_names[sample_offset]}",
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
                    support_maps=batch.S[sample_offset].detach().cpu(),
                    global_descriptor_score=float(scored.global_descriptor_score[sample_offset].item()),
                    provenance={
                        "pipeline_stage": "evaluation",
                        "score_mode": score_mode,
                        "timesteps_used": list(scored.timesteps_used),
                        "lambda_residual": float(lambda_residual if lambda_residual is not None else 0.0),
                        "lambda_noise": float(lambda_noise if lambda_noise is not None else 0.0),
                        "lambda_descriptor_spatial": float(
                            lambda_descriptor_spatial if lambda_descriptor_spatial is not None else descriptor_weight
                        ),
                        "lambda_descriptor_global": float(
                            lambda_descriptor_global if lambda_descriptor_global is not None else global_descriptor_weight
                        ),
                    },
                    source_paths={
                        "rgb_path": batch.rgb_paths[sample_offset],
                        "gt_mask_path": batch.gt_mask_paths[sample_offset],
                        "source_rgb_path": batch.source_rgb_paths[sample_offset],
                        "source_ir_path": batch.source_ir_paths[sample_offset],
                        "source_pointcloud_path": batch.source_pointcloud_paths[sample_offset],
                        "visualization_path": str(panel_path) if panel_path is not None else "",
                    },
                )
                package_path = evidence_dir / "packages" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.json"
                report_path = evidence_dir / "reports" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.md"
                save_evidence_package(package, package_path)
                save_templated_explanation(render_templated_explanation(package), report_path)

                llm_bundle_path = ""
                if enable_llm_explanations:
                    retrieved_context = retrieve_context_for_evidence(
                        package,
                        knowledge_base_root=knowledge_base_root,
                        top_k=retrieval_top_k,
                    )
                    prompt_payload = build_prompt_payload(package, retrieved_context)
                    explanation = build_llm_explanation(
                        package,
                        retrieved_context=retrieved_context,
                        prompt_payload=prompt_payload,
                    )
                    llm_bundle = evidence_dir / "llm" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.json"
                    llm_report = evidence_dir / "llm" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.md"
                    save_explanation_bundle(package, retrieved_context, prompt_payload, explanation, llm_bundle)
                    save_templated_explanation(render_llm_explanation_markdown(explanation), llm_report)
                    llm_bundle_path = str(llm_bundle)
                    llm_explanations_saved += 1

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
                        "global_descriptor_score": round(float(scored.global_descriptor_score[sample_offset].item()), 6),
                        "package_path": str(package_path),
                        "report_path": str(report_path),
                        "llm_bundle_path": llm_bundle_path,
                    }
                )

        reports.append(
            {
                "batch_index": batch_index,
                "mean_anomaly_score": round(float(scored.anomaly_score.mean().item()), 6),
                "mean_spatial_score": round(float(scored.spatial_score.mean().item()), 6),
                "mean_global_descriptor_score": round(float(scored.global_descriptor_score.mean().item()), 6),
                "max_anomaly_score": round(float(scored.anomaly_score.max().item()), 6),
                "score_mode": score_mode,
                "mean_score_basis": round(float(scored.score_basis_map.mean().item()), 6),
                "mean_residual": round(float(scored.residual_map.mean().item()), 6),
                "mean_noise_error": round(float(scored.noise_error_map.mean().item()), 6),
                "mean_descriptor_support": round(float(scored.descriptor_support.mean().item()), 6),
                "mean_residual_component": round(float(scored.residual_component.mean().item()), 6),
                "mean_noise_component": round(float(scored.noise_component.mean().item()), 6),
                "mean_descriptor_spatial_component": round(float(scored.descriptor_spatial_component.mean().item()), 6),
                "timesteps_used": list(scored.timesteps_used),
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
        "llm_explanations_saved": llm_explanations_saved,
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
    device_mode: str = "",
    learning_rate: float = 1e-4,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float = 0.0,
    lambda_noise: float = 1.0,
    lambda_descriptor_spatial: float = 0.25,
    lambda_descriptor_global: float = 0.1,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    base_channels: int = 32,
    global_embedding_dim: int = 128,
    time_embedding_dim: int = 128,
    attention_heads: int = 4,
    latent_channels: int = 4,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
    autoencoder_reconstruction_weight: float = 0.1,
    object_crop_enabled: bool = False,
    object_crop_margin_ratio: float = 0.10,
    object_crop_mask_source: str = "rgb_then_pc_density",
    rgb_normalization_mode: str = "none",
) -> Dict[str, object]:
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
    )

    train_dataset: DescriptorConditioningDataset = loaders["train_dataset"]  # type: ignore[assignment]
    if len(train_dataset) == 0:
        raise ValueError(f"No training records available for smoke experiment category={category!r}")

    model = build_model(
        descriptor_channels=train_dataset.descriptor_channels,
        global_dim=train_dataset.global_dim,
        base_channels=base_channels,
        global_embedding_dim=global_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        attention_heads=attention_heads,
        latent_channels=latent_channels,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        autoencoder_reconstruction_weight=autoencoder_reconstruction_weight,
    )
    runtime = resolve_runtime(
        requested_mode=_resolve_requested_mode(device, device_mode),
        model=model,
        validation_batch=prepare_diffusion_batch(next(iter(loaders["train_loader"]))),  # type: ignore[arg-type]
        run_backward=True,
    )
    torch_device = runtime.resolved_device
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    batch_reports = run_training_epoch(
        model,
        loaders["train_loader"],  # type: ignore[arg-type]
        optimizer,
        device=torch_device,
        use_amp=runtime.use_amp,
        epoch=1,
        max_batches=max_batches,
        log_every_n_steps=0,
        log_to_jsonl=False,
    )
    global_descriptor_calibration = fit_global_descriptor_reference(
        loaders["train_loader"],  # type: ignore[arg-type]
        category=category,
    )
    eval_reports = run_eval_scoring(
        model,
        loaders["eval_loader"],  # type: ignore[arg-type]
        device=torch_device,
        target_size=target_size,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
        multi_timestep_fractions=multi_timestep_fractions,
        anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
        lambda_residual=lambda_residual,
        lambda_noise=lambda_noise,
        lambda_descriptor_spatial=lambda_descriptor_spatial,
        lambda_descriptor_global=lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        pixel_threshold_percentile=pixel_threshold_percentile,
        object_mask_threshold=object_mask_threshold,
        max_batches=1,
        enable_llm_explanations=False,
        rgb_normalization_stats=loaders["rgb_normalization_stats"],  # type: ignore[arg-type]
    )

    manifests = loaders["manifests"]  # type: ignore[assignment]
    summary = {
        "category": category,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "target_size": target_size,
        "device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "anomaly_timestep": anomaly_timestep,
        "descriptor_weight": descriptor_weight,
        "global_descriptor_weight": global_descriptor_weight,
        "multi_timestep_scoring_enabled": multi_timestep_scoring_enabled,
        "anomaly_map_gaussian_sigma": anomaly_map_gaussian_sigma,
        "lambda_residual": lambda_residual,
        "lambda_noise": lambda_noise,
        "lambda_descriptor_spatial": lambda_descriptor_spatial,
        "lambda_descriptor_global": lambda_descriptor_global,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "object_mask_threshold": object_mask_threshold,
        "object_crop_enabled": object_crop_enabled,
        "object_crop_margin_ratio": object_crop_margin_ratio,
        "object_crop_mask_source": object_crop_mask_source,
        "rgb_normalization_mode": rgb_normalization_mode,
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
    epochs: int = 50,
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    device_mode: str = "",
    learning_rate: float = 1e-4,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool = True,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 3.0,
    lambda_residual: float = 1.0,
    lambda_noise: float = 0.5,
    lambda_descriptor_spatial: float = 0.25,
    lambda_descriptor_global: float = 0.3,
    score_weight_schedule: Dict[str, object] | None = None,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    max_visualizations: int = 6,
    base_channels: int = 32,
    global_embedding_dim: int = 128,
    time_embedding_dim: int = 128,
    attention_heads: int = 4,
    multiscale_conditioning: bool = True,
    latent_channels: int = 4,
    diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    noise_schedule: str = "cosine",
    autoencoder_reconstruction_weight: float = 0.1,
    object_crop_enabled: bool = False,
    object_crop_margin_ratio: float = 0.10,
    object_crop_mask_source: str = "rgb_then_pc_density",
    rgb_normalization_mode: str = "none",
    log_every_n_steps: int = 10,
    log_to_jsonl: bool = True,
) -> Dict[str, Any]:
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
    )

    train_dataset: DescriptorConditioningDataset = loaders["train_dataset"]  # type: ignore[assignment]
    eval_dataset: DescriptorConditioningDataset = loaders["eval_dataset"]  # type: ignore[assignment]
    if len(train_dataset) == 0:
        raise ValueError(f"No training records available for category={category!r}")
    rgb_stats: RGBNormalizationStats | None = loaders["rgb_normalization_stats"]  # type: ignore[assignment]
    if score_weight_schedule is None:
        score_weight_schedule = {
            "warmup_fraction": 0.4,
            "warmup": {
                "lambda_residual": 0.2,
                "lambda_noise": 0.3,
                "lambda_descriptor_spatial": 0.25,
                "lambda_descriptor_global": 0.8,
            },
            "main": {
                "lambda_residual": lambda_residual,
                "lambda_noise": lambda_noise,
                "lambda_descriptor_spatial": lambda_descriptor_spatial,
                "lambda_descriptor_global": lambda_descriptor_global,
            },
        }
    model = build_model(
        descriptor_channels=train_dataset.descriptor_channels,
        global_dim=train_dataset.global_dim,
        base_channels=base_channels,
        global_embedding_dim=global_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        attention_heads=attention_heads,
        latent_channels=latent_channels,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_schedule=noise_schedule,
        autoencoder_reconstruction_weight=autoencoder_reconstruction_weight,
    )
    runtime = resolve_runtime(
        requested_mode=_resolve_requested_mode(device, device_mode),
        model=model,
        validation_batch=prepare_diffusion_batch(next(iter(loaders["train_loader"]))),  # type: ignore[arg-type]
        run_backward=True,
    )
    torch_device = runtime.resolved_device
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    run_paths = create_run_dir(output_root=output_root, run_name=run_name, category=category)
    manifests = loaders["manifests"]  # type: ignore[assignment]
    rgb_stats_path = ""
    if rgb_stats is not None:
        rgb_stats_path = str(save_rgb_normalization_stats(rgb_stats, run_paths.metrics / "rgb_normalization_stats.json"))
    global_descriptor_calibration = fit_global_descriptor_reference(
        loaders["train_loader"],  # type: ignore[arg-type]
        category=category,
    )
    global_descriptor_calibration_path = write_json(
        run_paths.metrics / "global_descriptor_calibration.json",
        global_descriptor_calibration.to_dict(),
    )

    config_payload = {
        "category": category,
        "epochs": epochs,
        "batch_size": batch_size,
        "target_size": target_size,
        "device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "anomaly_timestep": anomaly_timestep,
        "descriptor_weight": descriptor_weight,
        "global_descriptor_weight": global_descriptor_weight,
        "multi_timestep_scoring_enabled": multi_timestep_scoring_enabled,
        "multi_timestep_fractions": list(multi_timestep_fractions or (0.05, 0.10, 0.20, 0.30, 0.40)),
        "anomaly_map_gaussian_sigma": anomaly_map_gaussian_sigma,
        "lambda_residual": lambda_residual,
        "lambda_noise": lambda_noise,
        "lambda_descriptor_spatial": lambda_descriptor_spatial,
        "lambda_descriptor_global": lambda_descriptor_global,
        "score_weight_schedule": score_weight_schedule,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "object_mask_threshold": object_mask_threshold,
        "object_crop_enabled": object_crop_enabled,
        "object_crop_margin_ratio": object_crop_margin_ratio,
        "object_crop_mask_source": object_crop_mask_source,
        "rgb_normalization_mode": rgb_normalization_mode,
        "rgb_normalization_stats": rgb_stats.to_dict() if rgb_stats is not None else None,
        "rgb_normalization_stats_path": rgb_stats_path,
        "training_protocol": "train_good_only",
        "evaluation_protocol": "test_good_plus_anomaly",
        "train_manifest_csv": str(manifests["train_csv"]),
        "eval_manifest_csv": str(manifests["eval_csv"]),
        "global_stats_json": str(manifests["global_stats_json"]),
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "base_channels": base_channels,
        "global_embedding_dim": global_embedding_dim,
        "time_embedding_dim": time_embedding_dim,
        "attention_heads": attention_heads,
        "multiscale_conditioning": multiscale_conditioning,
        "latent_channels": latent_channels,
        "diffusion_steps": diffusion_steps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "noise_schedule": noise_schedule,
        "autoencoder_reconstruction_weight": autoencoder_reconstruction_weight,
        "log_every_n_steps": log_every_n_steps,
        "log_to_jsonl": log_to_jsonl,
    }
    write_json(run_paths.root / "config.json", config_payload)

    history: List[Dict[str, Any]] = []
    best_auroc = float("-inf")
    best_checkpoint_path: Path | None = None
    step_log_path = run_paths.logs / "train_steps.jsonl"
    event_log_path = run_paths.logs / "events.jsonl"

    print(
        f"[train] category={category} epochs={epochs} device={torch_device} "
        f"train_records={len(train_dataset)} eval_records={len(eval_dataset)}"
    )
    append_jsonl(
        event_log_path,
        {
            "event": "run_started",
            "category": category,
            "epochs": epochs,
            "device_mode": runtime.requested_mode,
            "resolved_device": str(torch_device),
            "amp_enabled": runtime.use_amp,
            "fallback_reason": runtime.fallback_reason,
            "training_protocol": "train_good_only",
            "evaluation_protocol": "test_good_plus_anomaly",
        },
    )

    for epoch in range(1, epochs + 1):
        score_weights = resolve_score_weights(
            total_epochs=epochs,
            epoch=epoch,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            score_weight_schedule=score_weight_schedule,
        )
        train_reports = run_training_epoch(
            model,
            loaders["train_loader"],  # type: ignore[arg-type]
            optimizer,
            device=torch_device,
            use_amp=runtime.use_amp,
            epoch=epoch,
            max_batches=max_train_batches,
            seed=7 + epoch,
            log_every_n_steps=log_every_n_steps,
            log_to_jsonl=log_to_jsonl,
            step_log_path=step_log_path,
        )
        eval_summary = run_eval_scoring(
            model,
            loaders["eval_loader"],  # type: ignore[arg-type]
            device=torch_device,
            target_size=target_size,
            score_mode=score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=score_weights.lambda_residual,
            lambda_noise=score_weights.lambda_noise,
            lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
            lambda_descriptor_global=score_weights.lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            pixel_threshold_percentile=pixel_threshold_percentile,
            object_mask_threshold=object_mask_threshold,
            max_batches=max_eval_batches,
            visualization_dir=run_paths.predictions / f"epoch_{epoch:03d}",
            max_visualizations=max_visualizations,
            enable_llm_explanations=False,
            rgb_normalization_stats=rgb_stats,
            seed=17 + epoch,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(sum(report["loss"] for report in train_reports) / max(len(train_reports), 1), 6),
            "noise_loss": round(sum(report["noise_loss"] for report in train_reports) / max(len(train_reports), 1), 6),
            "autoencoder_reconstruction_loss": round(
                sum(report["autoencoder_reconstruction_loss"] for report in train_reports) / max(len(train_reports), 1),
                6,
            ),
            "image_auroc": eval_summary["image_auroc"],
            "image_f1": eval_summary["image_f1"],
            "pixel_f1": eval_summary["pixel_f1"],
            "pixel_iou": eval_summary["pixel_iou"],
            "mean_good_score": eval_summary["mean_good_score"],
            "mean_anomalous_score": eval_summary["mean_anomalous_score"],
            **score_weights.to_dict(),
        }
        history.append(epoch_summary)

        write_json(
            run_paths.metrics / f"epoch_{epoch:03d}.json",
            {
                "epoch": epoch,
                "train_reports": train_reports,
                "eval_summary": eval_summary,
            },
        )

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
            print(
                f"[train] epoch={epoch} new_best image_auroc={eval_summary['image_auroc']:.6f} "
                f"checkpoint={best_checkpoint_path}"
            )
            append_jsonl(
                event_log_path,
                {
                    "event": "new_best_checkpoint",
                    "epoch": epoch,
                    "image_auroc": eval_summary["image_auroc"],
                    "checkpoint_path": str(best_checkpoint_path),
                },
            )

        write_history_csv(run_paths.logs / "history.csv", history)
        write_json(run_paths.logs / "history.json", history)
        save_training_curves(history, run_paths.plots / "training_curves.png")
        append_jsonl(
            event_log_path,
            {
                "event": "epoch_completed",
                "epoch": epoch,
                "train_loss": epoch_summary["train_loss"],
                "image_auroc": epoch_summary["image_auroc"],
                "pixel_iou": epoch_summary["pixel_iou"],
                **score_weights.to_dict(),
            },
        )

    final_summary = {
        "run_dir": str(run_paths.root),
        "category": category,
        "epochs": epochs,
        "device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "train_records": len(train_dataset),
        "eval_records": len(eval_dataset),
        "training_protocol": "train_good_only",
        "evaluation_protocol": "test_good_plus_anomaly",
        "rgb_normalization_stats_path": rgb_stats_path,
        "best_image_auroc": round(best_auroc, 6) if best_auroc != float("-inf") else 0.0,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else "",
        "history_rows": len(history),
        "global_descriptor_calibration_path": str(global_descriptor_calibration_path),
    }
    write_json(run_paths.root / "summary.json", final_summary)
    append_jsonl(event_log_path, {"event": "run_completed", "summary": final_summary})
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
    device_mode: str = "",
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool | None = None,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float | None = None,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    max_eval_batches: int = 0,
    max_visualizations: int = 10,
    knowledge_base_root: Path | str | None = None,
    retrieval_top_k: int = 3,
    enable_llm_explanations: bool = True,
    object_crop_enabled: bool | None = None,
    object_crop_margin_ratio: float | None = None,
    object_crop_mask_source: str = "",
    rgb_normalization_mode: str = "",
) -> Dict[str, Any]:
    checkpoint_payload = read_checkpoint_payload(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint_config = checkpoint_payload.get("config", {})
    resolved_object_crop_enabled = (
        bool(object_crop_enabled)
        if object_crop_enabled is not None
        else bool(checkpoint_config.get("object_crop_enabled", False))
    )
    resolved_object_crop_margin_ratio = float(
        object_crop_margin_ratio
        if object_crop_margin_ratio is not None
        else checkpoint_config.get("object_crop_margin_ratio", 0.10)
    )
    resolved_object_crop_mask_source = object_crop_mask_source or str(
        checkpoint_config.get("object_crop_mask_source", "rgb_then_pc_density")
    )
    resolved_rgb_normalization_mode = rgb_normalization_mode or str(
        checkpoint_config.get("rgb_normalization_mode", "none")
    )
    resolved_rgb_stats = _coerce_rgb_normalization_stats(checkpoint_config.get("rgb_normalization_stats"))
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
        object_crop_enabled=resolved_object_crop_enabled,
        object_crop_margin_ratio=resolved_object_crop_margin_ratio,
        object_crop_mask_source=resolved_object_crop_mask_source,
        rgb_normalization_mode=resolved_rgb_normalization_mode,
        rgb_normalization_stats=resolved_rgb_stats,
    )
    eval_dataset: DescriptorConditioningDataset = loaders["eval_dataset"]  # type: ignore[assignment]
    rgb_stats: RGBNormalizationStats | None = loaders["rgb_normalization_stats"]  # type: ignore[assignment]

    model = build_model(
        descriptor_channels=eval_dataset.descriptor_channels,
        global_dim=eval_dataset.global_dim,
        base_channels=int(checkpoint_config.get("base_channels", 32)),
        global_embedding_dim=int(checkpoint_config.get("global_embedding_dim", 128)),
        time_embedding_dim=int(checkpoint_config.get("time_embedding_dim", 128)),
        attention_heads=int(checkpoint_config.get("attention_heads", 4)),
        latent_channels=int(checkpoint_config.get("latent_channels", 4)),
        diffusion_steps=int(checkpoint_config.get("diffusion_steps", 1000)),
        beta_start=float(checkpoint_config.get("beta_start", 1e-4)),
        beta_end=float(checkpoint_config.get("beta_end", 2e-2)),
        noise_schedule=str(checkpoint_config.get("noise_schedule", "cosine")),
        autoencoder_reconstruction_weight=float(
            checkpoint_config.get("autoencoder_reconstruction_weight", 0.1)
        ),
    )
    runtime = resolve_runtime(
        requested_mode=_resolve_requested_mode(device, device_mode),
        model=model,
        validation_batch=prepare_diffusion_batch(next(iter(loaders["eval_loader"]))),  # type: ignore[arg-type]
        run_backward=False,
    )
    torch_device = runtime.resolved_device
    payload = load_checkpoint(checkpoint_path, model=model, map_location=torch_device)
    resolved_score_mode = score_mode or str(checkpoint_config.get("score_mode", "noise_error"))
    resolved_multi_timestep = (
        bool(multi_timestep_scoring_enabled)
        if multi_timestep_scoring_enabled is not None
        else bool(checkpoint_config.get("multi_timestep_scoring_enabled", False))
    )
    resolved_multi_timestep_fractions = (
        list(multi_timestep_fractions)
        if multi_timestep_fractions is not None
        else list(checkpoint_config.get("multi_timestep_fractions", [0.05, 0.10, 0.20, 0.30, 0.40]))
    )
    resolved_anomaly_map_sigma = float(
        anomaly_map_gaussian_sigma
        if anomaly_map_gaussian_sigma is not None
        else checkpoint_config.get("anomaly_map_gaussian_sigma", 0.0)
    )
    score_weights = resolve_score_weights(
        total_epochs=int(checkpoint_config.get("epochs", 0)),
        epoch=int(payload.get("epoch", 0)) or int(checkpoint_config.get("epochs", 0) or 0),
        lambda_residual=float(
            lambda_residual if lambda_residual is not None else checkpoint_config.get("lambda_residual", 0.0)
        ),
        lambda_noise=float(
            lambda_noise if lambda_noise is not None else checkpoint_config.get("lambda_noise", 1.0)
        ),
        lambda_descriptor_spatial=float(
            lambda_descriptor_spatial
            if lambda_descriptor_spatial is not None
            else checkpoint_config.get("lambda_descriptor_spatial", descriptor_weight)
        ),
        lambda_descriptor_global=float(
            lambda_descriptor_global
            if lambda_descriptor_global is not None
            else checkpoint_config.get("lambda_descriptor_global", global_descriptor_weight)
        ),
        score_weight_schedule=checkpoint_config.get("score_weight_schedule"),
    )

    run_paths = create_run_dir(output_root=output_root, run_name=run_name or "eval", category=category)
    evidence_root = run_paths.root / "evidence"
    rgb_stats_path = ""
    if rgb_stats is not None:
        rgb_stats_path = str(save_rgb_normalization_stats(rgb_stats, evidence_root / "rgb_normalization_stats.json"))
    global_descriptor_calibration = fit_global_descriptor_reference(
        loaders["train_loader"],  # type: ignore[arg-type]
        category=category,
    )
    global_descriptor_calibration_path = write_json(
        evidence_root / "global_descriptor_calibration.json",
        global_descriptor_calibration.to_dict(),
    )
    calibration_scores = collect_reference_scores(
        model,
        loaders["train_loader"],  # type: ignore[arg-type]
        device=torch_device,
        score_mode=resolved_score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=resolved_multi_timestep,
        multi_timestep_fractions=resolved_multi_timestep_fractions,
        anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
        lambda_residual=score_weights.lambda_residual,
        lambda_noise=score_weights.lambda_noise,
        lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
        lambda_descriptor_global=score_weights.lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        object_mask_threshold=object_mask_threshold,
    )
    calibration = fit_masi_calibration(calibration_scores, category=category, score_mode=resolved_score_mode)
    calibration_path = write_json(evidence_root / "calibration.json", calibration.to_dict())
    eval_summary = run_eval_scoring(
        model,
        loaders["eval_loader"],  # type: ignore[arg-type]
        device=torch_device,
        target_size=target_size,
        score_mode=resolved_score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=resolved_multi_timestep,
        multi_timestep_fractions=resolved_multi_timestep_fractions,
        anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
        lambda_residual=score_weights.lambda_residual,
        lambda_noise=score_weights.lambda_noise,
        lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
        lambda_descriptor_global=score_weights.lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        pixel_threshold_percentile=pixel_threshold_percentile,
        object_mask_threshold=object_mask_threshold,
        max_batches=max_eval_batches,
        visualization_dir=run_paths.predictions / "eval",
        evidence_dir=evidence_root,
        calibration=calibration,
        max_visualizations=max_visualizations,
        knowledge_base_root=Path(knowledge_base_root) if knowledge_base_root else None,
        retrieval_top_k=retrieval_top_k,
        enable_llm_explanations=enable_llm_explanations,
        rgb_normalization_stats=rgb_stats,
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "loaded_epoch": int(payload.get("epoch", 0)),
        "category": category,
        "device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "score_mode": resolved_score_mode,
        "eval_records": len(eval_dataset),
        "training_protocol": "train_good_only",
        "evaluation_protocol": "test_good_plus_anomaly",
        "image_auroc": eval_summary["image_auroc"],
        "image_f1": eval_summary["image_f1"],
        "pixel_f1": eval_summary["pixel_f1"],
        "pixel_iou": eval_summary["pixel_iou"],
        "mean_good_score": eval_summary["mean_good_score"],
        "mean_anomalous_score": eval_summary["mean_anomalous_score"],
        **score_weights.to_dict(),
        "visualizations_saved": eval_summary["visualizations_saved"],
        "evidence_packages_saved": eval_summary["evidence_packages_saved"],
        "llm_explanations_saved": eval_summary["llm_explanations_saved"],
        "calibration_path": str(calibration_path),
        "global_descriptor_calibration_path": str(global_descriptor_calibration_path),
        "rgb_normalization_stats_path": rgb_stats_path,
        "evidence_index_path": eval_summary["evidence_index_path"],
        "run_dir": str(run_paths.root),
    }
    write_json(run_paths.metrics / "evaluation.json", eval_summary)
    write_json(run_paths.root / "summary.json", summary)
    return {"run_paths": run_paths, "summary": summary, "eval_summary": eval_summary}


def fit_global_descriptor_reference(loader: DataLoader, *, category: str, max_batches: int = 0) -> GlobalDescriptorCalibration:
    vectors = collect_reference_global_vectors(loader, max_batches=max_batches)
    return fit_global_descriptor_calibration(
        vectors,
        category=category,
        feature_names=GLOBAL_FEATURE_NAMES,
    )


def collect_reference_global_vectors(loader: DataLoader, *, max_batches: int = 0) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        collected.append(batch.g.detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break
    return torch.cat(collected) if collected else torch.zeros((1, len(GLOBAL_FEATURE_NAMES)), dtype=torch.float32)


def collect_reference_scores(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | None,
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
            batch.S.to(device),
            batch.g.to(device),
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        collected.append(scored.anomaly_score.detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break

    return torch.cat(collected) if collected else torch.empty(0)


def load_gt_mask(
    path_like: str,
    *,
    target_size: tuple[int, int],
    crop_box_xyxy: tuple[int, int, int, int] | None = None,
    crop_source_size_hw: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    path = Path(path_like) if path_like else None
    if not path or not path.exists():
        return None
    image = Image.open(path).convert("L")
    if crop_box_xyxy is not None and crop_source_size_hw is not None:
        source_h, source_w = crop_source_size_hw
        image_w, image_h = image.size
        scale_x = image_w / max(source_w, 1)
        scale_y = image_h / max(source_h, 1)
        left, top, right, bottom = crop_box_xyxy
        image = image.crop(
            (
                int(round(left * scale_x)),
                int(round(top * scale_y)),
                int(round(right * scale_x)),
                int(round(bottom * scale_y)),
            )
        )
    image = TF.resize(image, list(target_size), interpolation=InterpolationMode.NEAREST)
    tensor = TF.pil_to_tensor(image).float() / 255.0
    return (tensor > 0.5).float()
