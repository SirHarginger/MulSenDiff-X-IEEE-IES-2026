from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from src.category_policies import is_archetype_a_replace
from src.project_layout import resolve_processed_root
from src.preprocessing.dataset import MODALITY_SPECS
from src.preprocessing.ir_descriptors import compute_ir_category_stats, save_ir_category_stats
from src.preprocessing.pointcloud_descriptors import (
    compute_category_bbox_stats,
    compute_pointcloud_feature_maps,
    save_pointcloud_category_stats,
)


def build_category_stats(
    *,
    raw_root: Path | str,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    categories: Sequence[str] | None = None,
    target_size: tuple[int, int] = (256, 256),
    log_progress: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    raw_root = Path(raw_root)
    data_root = Path(data_root)
    resolved_processed_root = resolve_processed_root(data_root=data_root, processed_root=processed_root)
    categories_filter = {item for item in (categories or []) if item}
    stats_root = resolved_processed_root / "category_stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    built_categories: list[str] = []
    for category_path in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        category = category_path.name
        if categories_filter and category not in categories_filter:
            continue

        ir_paths = sorted((category_path / "Infrared" / "train").glob("*.png"))
        pointcloud_paths = sorted((category_path / "Pointcloud" / "train").glob("*.stl"))
        if not ir_paths or not pointcloud_paths:
            continue

        _log(
            log_progress,
            f"category_stats:{category}: start ({len(ir_paths)} IR train images, {len(pointcloud_paths)} point clouds)",
        )
        out_dir = stats_root / category
        out_dir.mkdir(parents=True, exist_ok=True)

        _log(log_progress, f"category_stats:{category}: computing IR stats")
        ir_stats = compute_ir_category_stats(ir_paths, target_size=target_size)
        save_ir_category_stats(ir_stats, out_dir)

        _log(log_progress, f"category_stats:{category}: computing point-cloud bounds")
        bbox_stats = compute_category_bbox_stats(pointcloud_paths)
        depth_maps: list[np.ndarray] = []
        roughness_maps: list[np.ndarray] = []
        curvature_maps: list[np.ndarray] = []
        normal_maps: list[np.ndarray] = []
        archetype_a_depth_reference_samples: list[np.ndarray] = []
        archetype_a_normal_reference_samples: list[np.ndarray] = []
        feature_maps_by_path: dict[str, Dict[str, np.ndarray]] = {}
        density_max = 1.0
        descriptor_policy = "residual_reference" if is_archetype_a_replace(category) else "default_geometry"

        _log(log_progress, f"category_stats:{category}: point-cloud feature pass 1/2")
        for index, path in enumerate(pointcloud_paths, start=1):
            if _should_log_sample_progress(index, len(pointcloud_paths)):
                _log(log_progress, f"category_stats:{category}: pass 1/2 point cloud {index}/{len(pointcloud_paths)}")
            feature_maps = compute_pointcloud_feature_maps(
                path,
                stats={**bbox_stats, "normal_deviation_mean": np.zeros((*target_size, 3), dtype=np.float32)},
                projection_size=target_size,
            )
            feature_maps_by_path[str(path)] = feature_maps
            depth_maps.append(feature_maps["depth"])
            roughness_maps.append(feature_maps["roughness"])
            curvature_maps.append(feature_maps["curvature"])
            normal_maps.append(feature_maps["normal_map"])
            density_max = max(density_max, float(feature_maps["density"].max()))
            if is_archetype_a_replace(category):
                archetype_a_depth_reference_samples.append(feature_maps["depth"])
                archetype_a_normal_reference_samples.append(feature_maps["normal_map"])

        depth_mean = np.stack(depth_maps, axis=0).mean(axis=0).astype(np.float32)
        roughness_mean = np.stack(roughness_maps, axis=0).mean(axis=0).astype(np.float32)
        curvature_mean = np.stack(curvature_maps, axis=0).mean(axis=0).astype(np.float32)
        normal_mean = np.stack(normal_maps, axis=0).mean(axis=0).astype(np.float32)
        normal_norm = np.linalg.norm(normal_mean, axis=2, keepdims=True)
        normal_mean = normal_mean / np.clip(normal_norm, 1e-8, None)

        pointcloud_stats = {
            **bbox_stats,
            "normal_deviation_mean": normal_mean,
        }

        depth_reference_std = None
        normal_reference_std = None
        if is_archetype_a_replace(category) and archetype_a_depth_reference_samples and archetype_a_normal_reference_samples:
            depth_reference_std = np.stack(archetype_a_depth_reference_samples, axis=0).std(axis=0).astype(np.float32)
            normal_reference_std = np.stack(archetype_a_normal_reference_samples, axis=0).std(axis=0).astype(np.float32)

        depth_residual_min = float("inf")
        depth_residual_max = float("-inf")
        roughness_residual_max = 0.0
        curvature_residual_max = 0.0
        _log(log_progress, f"category_stats:{category}: point-cloud residual pass 2/2")
        for index, path in enumerate(pointcloud_paths, start=1):
            if _should_log_sample_progress(index, len(pointcloud_paths)):
                _log(log_progress, f"category_stats:{category}: pass 2/2 point cloud {index}/{len(pointcloud_paths)}")
            feature_maps = feature_maps_by_path[str(path)]
            depth_residual = feature_maps["depth"] - depth_mean
            depth_residual_min = min(depth_residual_min, float(depth_residual.min()))
            depth_residual_max = max(depth_residual_max, float(depth_residual.max()))
            roughness_residual = np.clip(feature_maps["roughness"] - roughness_mean, 0.0, None)
            curvature_residual = np.clip(feature_maps["curvature"] - curvature_mean, 0.0, None)
            roughness_residual_max = max(roughness_residual_max, float(roughness_residual.max()))
            curvature_residual_max = max(curvature_residual_max, float(curvature_residual.max()))

        save_pointcloud_category_stats(
            pointcloud_stats,
            out_dir,
            depth_mean=depth_mean,
            roughness_mean=roughness_mean,
            curvature_mean=curvature_mean,
            normal_deviation_mean=normal_mean,
            density_max=density_max,
            depth_residual_range=(depth_residual_min, depth_residual_max),
            roughness_residual_max=roughness_residual_max,
            curvature_residual_max=curvature_residual_max,
            descriptor_policy=descriptor_policy,
            depth_reference_std=depth_reference_std,
            normal_reference_std=normal_reference_std,
        )

        _log(log_progress, f"category_stats:{category}: saved stats to {out_dir}")
        built_categories.append(category)

    return {
        "built_categories": built_categories,
        "stats_root": stats_root,
    }


def _log(log_progress: Callable[[str], None] | None, message: str) -> None:
    if log_progress is not None:
        log_progress(message)


def _should_log_sample_progress(index: int, total: int) -> bool:
    if total <= 10:
        return True
    if index in {1, total}:
        return True
    return index % 25 == 0
