from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.project_layout import resolve_processed_root


GLOBAL_FEATURE_NAMES = [
    "ir_hotspot_area_fraction",
    "ir_max_gradient",
    "ir_mean_hotspot_intensity",
    "ir_hotspot_compactness",
    "pc_mean_curvature",
    "pc_max_roughness",
    "pc_p95_normal_deviation",
    "cross_overlap_score",
    "cross_centroid_distance",
    "cross_inconsistency_score",
]

SPATIAL_DESCRIPTOR_ORDER = [
    "ir_hotspot",
    "ir_gradient",
    "ir_variance",
    "pc_curvature",
    "pc_roughness",
    "pc_normal_deviation",
    "cross_agreement",
]

SUPPORT_DESCRIPTOR_ORDER = [
    "ir_normalized",
    "pc_depth",
    "pc_density",
    "cross_ir_support",
    "cross_geo_support",
    "cross_inconsistency",
]

SYNTHETIC_VALIDATION_RECIPES = (
    "thermal_hotspot",
    "geometric_surface_defect",
    "cross_modal_inconsistency",
)

_REQUIRED_SAMPLE_FILES = [
    "rgb.png",
    "global.json",
    "meta.json",
    *[f"{name}.png" for name in SPATIAL_DESCRIPTOR_ORDER],
    *[f"{name}.png" for name in SUPPORT_DESCRIPTOR_ORDER],
]


@dataclass(frozen=True)
class ProcessedSampleRecord:
    category: str
    split: str
    defect_label: str
    sample_id: str
    sample_name: str
    sample_dir: str
    is_anomalous: bool
    gt_mask_path: str
    rgb_source_path: str
    ir_source_path: str
    pc_source_path: str


@dataclass(frozen=True)
class RGBNormalizationStats:
    category: str
    mode: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    masked_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalVectorNormalizationStats:
    category: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["mean"] = list(self.mean)
        payload["std"] = list(self.std)
        return payload


@dataclass(frozen=True)
class DescriptorConditioning:
    descriptor_maps: torch.Tensor
    support_maps: torch.Tensor
    global_vector: torch.Tensor
    raw_global_vector: torch.Tensor


@dataclass(frozen=True)
class ModelSample:
    x_rgb: torch.Tensor
    conditioning: DescriptorConditioning
    category: str
    category_index: int
    split: str
    defect_label: str
    sample_id: str
    sample_name: str
    is_anomalous: bool
    sample_dir: str
    rgb_path: str
    gt_mask_path: str
    source_rgb_path: str
    source_ir_path: str
    source_pointcloud_path: str
    crop_box_xyxy: tuple[int, int, int, int] | None
    crop_source_size_hw: tuple[int, int]
    rgb_normalization_mode: str = "none"


@dataclass(frozen=True)
class ObjectCropConfig:
    enabled: bool = False
    margin_ratio: float = 0.10
    mask_source: str = "rgb_then_pc_density"
    brightness_threshold: float = 0.02
    pc_density_threshold: float = 0.05
    min_mask_pixels: int = 16


def _normalize_category_list(
    categories: Sequence[str] | None,
    *,
    samples_root: Path,
) -> list[str]:
    available = sorted({row.category for row in iter_processed_sample_manifest(samples_root)})
    requested = [item.strip() for item in (categories or []) if str(item).strip()]
    if not requested:
        return available
    if len(requested) == 1 and requested[0].lower() == "all":
        return available
    requested_set = set(requested)
    return [category for category in available if category in requested_set]


def _bool_from_string(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return TF.pil_to_tensor(image).float() / 255.0


def _load_gray_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    return TF.pil_to_tensor(image).float() / 255.0


def _resize_tensor(tensor: torch.Tensor, target_size: tuple[int, int], *, nearest: bool = False) -> torch.Tensor:
    mode = "nearest" if nearest else "bilinear"
    align_corners = None if nearest else False
    return F.interpolate(
        tensor.unsqueeze(0),
        size=target_size,
        mode=mode,
        align_corners=align_corners,
    ).squeeze(0)


def _crop_tensor(tensor: torch.Tensor, crop_box_xyxy: tuple[int, int, int, int] | None) -> torch.Tensor:
    if crop_box_xyxy is None:
        return tensor
    left, top, right, bottom = crop_box_xyxy
    return tensor[:, top:bottom, left:right]


def _mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys, xs = torch.nonzero(mask > 0, as_tuple=True)
    if xs.numel() == 0 or ys.numel() == 0:
        return None
    left = int(xs.min().item())
    right = int(xs.max().item()) + 1
    top = int(ys.min().item())
    bottom = int(ys.max().item()) + 1
    return left, top, right, bottom


def _expand_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox_xyxy
    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    pad_x = max(int(round(box_width * margin_ratio)), 1)
    pad_y = max(int(round(box_height * margin_ratio)), 1)
    return (
        max(left - pad_x, 0),
        max(top - pad_y, 0),
        min(right + pad_x, width),
        min(bottom + pad_y, height),
    )


def _build_crop_mask(
    rgb: torch.Tensor,
    pc_density: torch.Tensor,
    *,
    config: ObjectCropConfig,
) -> torch.Tensor:
    brightness = rgb.mean(dim=0)
    rgb_mask = brightness > config.brightness_threshold
    pc_mask = pc_density.squeeze(0) > config.pc_density_threshold

    if config.mask_source == "rgb":
        mask = rgb_mask
    elif config.mask_source == "pc_density":
        mask = pc_mask
    else:
        mask = rgb_mask if int(rgb_mask.sum().item()) >= config.min_mask_pixels else pc_mask
    if int(mask.sum().item()) < config.min_mask_pixels:
        mask = rgb_mask | pc_mask
    return mask.float()


def estimate_object_crop_box(
    rgb: torch.Tensor,
    pc_density: torch.Tensor,
    *,
    config: ObjectCropConfig,
) -> tuple[int, int, int, int] | None:
    if not config.enabled:
        return None
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    mask = _build_crop_mask(rgb, pc_density, config=config)
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    return _expand_bbox(bbox, width=width, height=height, margin_ratio=config.margin_ratio)


def estimate_object_mask_from_sample(
    rgb: torch.Tensor,
    support_maps: torch.Tensor,
    *,
    crop_config: ObjectCropConfig,
) -> torch.Tensor:
    pc_density = support_maps[SUPPORT_DESCRIPTOR_ORDER.index("pc_density") : SUPPORT_DESCRIPTOR_ORDER.index("pc_density") + 1]
    return _build_crop_mask(rgb, pc_density, config=crop_config).unsqueeze(0)


def _load_global_stats(path: Path | str | None) -> tuple[torch.Tensor, torch.Tensor]:
    if path is None:
        return (
            torch.zeros(len(GLOBAL_FEATURE_NAMES), dtype=torch.float32),
            torch.ones(len(GLOBAL_FEATURE_NAMES), dtype=torch.float32),
        )
    stats_path = Path(path)
    if not stats_path.exists():
        return (
            torch.zeros(len(GLOBAL_FEATURE_NAMES), dtype=torch.float32),
            torch.ones(len(GLOBAL_FEATURE_NAMES), dtype=torch.float32),
        )
    payload = _read_json(stats_path)
    feature_names = payload.get("feature_names", GLOBAL_FEATURE_NAMES)
    means_payload = payload.get("means", {})
    stds_payload = payload.get("stds", {})
    means = [float(means_payload.get(name, 0.0)) for name in feature_names]
    stds = [max(float(stds_payload.get(name, 1.0)), 1e-6) for name in feature_names]
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)


def _coerce_rgb_normalization_stats_payload(payload: Mapping[str, object]) -> RGBNormalizationStats:
    return RGBNormalizationStats(
        category=str(payload.get("category", "")),
        mode=str(payload.get("mode", "none")),
        mean=tuple(float(value) for value in payload.get("mean", [0.0, 0.0, 0.0])),
        std=tuple(max(float(value), 1e-6) for value in payload.get("std", [1.0, 1.0, 1.0])),
        masked_pixels=int(payload.get("masked_pixels", 0)),
    )


def _coerce_global_vector_normalization_stats_payload(
    payload: Mapping[str, object],
) -> GlobalVectorNormalizationStats:
    feature_names = tuple(str(value) for value in payload.get("feature_names", GLOBAL_FEATURE_NAMES))
    mean = tuple(float(value) for value in payload.get("mean", [0.0] * len(feature_names)))
    std = tuple(max(float(value), 1e-6) for value in payload.get("std", [1.0] * len(feature_names)))
    return GlobalVectorNormalizationStats(
        category=str(payload.get("category", "")),
        feature_names=feature_names,
        mean=mean,
        std=std,
        sample_count=int(payload.get("sample_count", 0)),
    )


def _load_rgb_normalization_stats(path: Path | str | None) -> RGBNormalizationStats | None:
    if path is None:
        return None
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    payload = _read_json(stats_path)
    if str(payload.get("mode", "")).lower() == "per_category" and isinstance(payload.get("categories"), dict):
        raise ValueError("Expected single-category RGB normalization stats, found per-category payload.")
    return _coerce_rgb_normalization_stats_payload(payload)


def _load_rgb_normalization_stats_by_category(
    path: Path | str | None,
) -> Dict[str, RGBNormalizationStats] | None:
    if path is None:
        return None
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    payload = _read_json(stats_path)
    if isinstance(payload.get("categories"), dict):
        return {
            str(category): _coerce_rgb_normalization_stats_payload(stats_payload)
            for category, stats_payload in payload["categories"].items()
            if isinstance(stats_payload, dict)
        }
    stats = _coerce_rgb_normalization_stats_payload(payload)
    return {stats.category: stats} if stats.category else None


def _load_global_vector_normalization_stats_by_category(
    path: Path | str | None,
) -> Dict[str, GlobalVectorNormalizationStats] | None:
    if path is None:
        return None
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    payload = _read_json(stats_path)
    if isinstance(payload.get("categories"), dict):
        return {
            str(category): _coerce_global_vector_normalization_stats_payload(stats_payload)
            for category, stats_payload in payload["categories"].items()
            if isinstance(stats_payload, dict)
        }
    stats = _coerce_global_vector_normalization_stats_payload(payload)
    return {stats.category: stats} if stats.category else None


def _normalize_rgb(rgb: torch.Tensor, stats: RGBNormalizationStats | None) -> torch.Tensor:
    if stats is None or stats.mode == "none":
        return rgb
    mean = torch.tensor(stats.mean, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(stats.std, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return (rgb - mean) / std.clamp_min(1e-6)


def _normalize_global_vector(
    vector: torch.Tensor,
    stats: GlobalVectorNormalizationStats | None,
    *,
    fallback_means: torch.Tensor,
    fallback_stds: torch.Tensor,
) -> torch.Tensor:
    if stats is None:
        return (vector - fallback_means) / fallback_stds.clamp_min(1e-6)
    mean = torch.tensor(stats.mean, dtype=vector.dtype, device=vector.device)
    std = torch.tensor(stats.std, dtype=vector.dtype, device=vector.device)
    return (vector - mean) / std.clamp_min(1e-6)


def _sample_is_complete(sample_dir: Path) -> bool:
    return all((sample_dir / filename).exists() for filename in _REQUIRED_SAMPLE_FILES)


def iter_processed_sample_manifest(samples_root: Path | str) -> Iterator[ProcessedSampleRecord]:
    samples_root = Path(samples_root)
    if not samples_root.exists():
        return
    for sample_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        if not _sample_is_complete(sample_dir):
            continue
        meta = _read_json(sample_dir / "meta.json")
        yield ProcessedSampleRecord(
            category=str(meta.get("category", "")),
            split=str(meta.get("split", "")),
            defect_label=str(meta.get("defect_label", "")),
            sample_id=str(meta.get("sample_id", "")),
            sample_name=str(meta.get("sample_name", sample_dir.name)),
            sample_dir=str(sample_dir),
            is_anomalous=str(meta.get("defect_label", "")) not in {"", "good"},
            gt_mask_path=str(meta.get("gt_mask_path", "") or ""),
            rgb_source_path=str(meta.get("rgb_source_path", "")),
            ir_source_path=str(meta.get("ir_source_path", "")),
            pc_source_path=str(meta.get("pc_source_path", "")),
        )


def select_processed_sample_records(
    *,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    categories: Sequence[str] | None = None,
) -> List[ProcessedSampleRecord]:
    data_root = Path(data_root)
    samples_root = resolve_processed_root(data_root=data_root, processed_root=processed_root) / "samples"
    selected_categories = set(_normalize_category_list(categories, samples_root=samples_root))
    rows = list(iter_processed_sample_manifest(samples_root))
    if not selected_categories:
        return rows
    return [row for row in rows if row.category in selected_categories]


def _stable_sample_hash(sample_name: str) -> int:
    digest = hashlib.sha256(sample_name.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _split_train_core_and_calibration(
    rows: Sequence[ProcessedSampleRecord],
    *,
    calibration_fraction: float = 0.20,
) -> tuple[list[ProcessedSampleRecord], list[ProcessedSampleRecord]]:
    grouped: Dict[str, List[ProcessedSampleRecord]] = {}
    for row in rows:
        grouped.setdefault(row.category, []).append(row)

    train_core_rows: list[ProcessedSampleRecord] = []
    calibration_rows: list[ProcessedSampleRecord] = []
    for category, category_rows in sorted(grouped.items()):
        ordered = sorted(category_rows, key=lambda item: (_stable_sample_hash(item.sample_name), item.sample_name))
        if len(ordered) <= 1:
            train_core_rows.extend(ordered)
            continue
        calibration_count = max(1, int(round(len(ordered) * calibration_fraction)))
        calibration_count = min(calibration_count, len(ordered) - 1)
        calibration_rows.extend(ordered[:calibration_count])
        train_core_rows.extend(ordered[calibration_count:])
    return train_core_rows, calibration_rows


def _resolve_sample_dir(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else Path.cwd() / path


def _synthetic_mask(size_hw: tuple[int, int], *, sample_name: str, recipe: str) -> torch.Tensor:
    height, width = size_hw
    seed_value = _stable_sample_hash(f"{sample_name}:{recipe}") & 0x7FFFFFFF
    generator = torch.Generator()
    generator.manual_seed(seed_value)
    center_x = int(
        torch.randint(
            low=max(width // 4, 0),
            high=max((3 * width) // 4, width // 4 + 1),
            size=(1,),
            generator=generator,
        ).item()
    )
    center_y = int(
        torch.randint(
            low=max(height // 4, 0),
            high=max((3 * height) // 4, height // 4 + 1),
            size=(1,),
            generator=generator,
        ).item()
    )
    sigma = max(min(height, width) // 8, 2)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(2.0 * sigma * sigma, 1e-6))
    return mask.clamp(0.0, 1.0)


def _load_gray_stack(sample_dir: Path, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {
        name: _load_gray_tensor(sample_dir / f"{name}.png").squeeze(0)
        for name in names
    }


def _save_rgb_tensor(rgb: torch.Tensor, path: Path) -> None:
    array = (rgb.detach().clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path)


def _save_gray_stack(output_dir: Path, stack: Mapping[str, torch.Tensor]) -> None:
    for name, tensor in stack.items():
        array = (tensor.detach().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
        Image.fromarray(array, mode="L").save(output_dir / f"{name}.png")


def _apply_synthetic_recipe(
    recipe: str,
    descriptor_maps: Dict[str, torch.Tensor],
    support_maps: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    updated_descriptors = {key: value.clone() for key, value in descriptor_maps.items()}
    updated_support = {key: value.clone() for key, value in support_maps.items()}

    def bump(name: str, amount: float, *, support: bool = False) -> None:
        target = updated_support if support else updated_descriptors
        target[name] = (target[name] + amount * mask).clamp(0.0, 1.0)

    if recipe == "thermal_hotspot":
        bump("ir_hotspot", 0.75)
        bump("ir_gradient", 0.45)
        bump("ir_variance", 0.35)
        bump("cross_agreement", 0.25)
        bump("cross_ir_support", 0.50, support=True)
    elif recipe == "geometric_surface_defect":
        bump("pc_curvature", 0.55)
        bump("pc_roughness", 0.65)
        bump("pc_normal_deviation", 0.45)
        bump("cross_agreement", 0.20)
        bump("cross_geo_support", 0.50, support=True)
    elif recipe == "cross_modal_inconsistency":
        bump("ir_hotspot", 0.45)
        bump("cross_agreement", -0.25)
        bump("cross_inconsistency", 0.70, support=True)
        bump("cross_ir_support", 0.25, support=True)
    else:
        raise ValueError(f"Unsupported synthetic recipe {recipe!r}.")

    return updated_descriptors, updated_support


def _render_synthetic_rgb(rgb: torch.Tensor, mask: torch.Tensor, recipe: str) -> torch.Tensor:
    rendered = rgb.clone()
    if recipe == "thermal_hotspot":
        rendered[0] = (rendered[0] + 0.35 * mask).clamp(0.0, 1.0)
        rendered[1] = (rendered[1] + 0.15 * mask).clamp(0.0, 1.0)
    elif recipe == "geometric_surface_defect":
        rendered[0] = (rendered[0] - 0.20 * mask).clamp(0.0, 1.0)
        rendered[1] = (rendered[1] - 0.15 * mask).clamp(0.0, 1.0)
        rendered[2] = (rendered[2] - 0.10 * mask).clamp(0.0, 1.0)
    else:
        rendered[2] = (rendered[2] + 0.30 * mask).clamp(0.0, 1.0)
        rendered[0] = (rendered[0] - 0.10 * mask).clamp(0.0, 1.0)
    return rendered


def _mask_compactness(mask: torch.Tensor) -> float:
    binary = (mask > 0).to(dtype=torch.uint8)
    if int(binary.sum().item()) == 0:
        return 0.0
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1))
    center = padded[1:-1, 1:-1]
    perimeter = int(((center == 1) & (padded[:-2, 1:-1] == 0)).sum().item())
    perimeter += int(((center == 1) & (padded[2:, 1:-1] == 0)).sum().item())
    perimeter += int(((center == 1) & (padded[1:-1, :-2] == 0)).sum().item())
    perimeter += int(((center == 1) & (padded[1:-1, 2:] == 0)).sum().item())
    if perimeter <= 0:
        return 1.0
    area = float(binary.sum().item())
    return float((4.0 * 3.141592653589793 * area) / float(perimeter * perimeter))


def _top_percent_centroid_distance(left: torch.Tensor, right: torch.Tensor, *, percentile: float = 95.0) -> float:
    def centroid(array: torch.Tensor) -> tuple[float, float] | None:
        flat = array.detach().float()
        if flat.numel() == 0:
            return None
        threshold = torch.quantile(flat.reshape(-1), percentile / 100.0)
        coords = torch.nonzero(flat >= threshold, as_tuple=False)
        if coords.numel() == 0:
            return None
        ys = coords[:, 0].float()
        xs = coords[:, 1].float()
        return float(xs.mean().item()), float(ys.mean().item())

    left_centroid = centroid(left)
    right_centroid = centroid(right)
    if left_centroid is None or right_centroid is None:
        return 1.0
    height, width = left.shape
    diagonal = max(float((height * height + width * width) ** 0.5), 1e-6)
    dx = left_centroid[0] - right_centroid[0]
    dy = left_centroid[1] - right_centroid[1]
    return float(((dx * dx + dy * dy) ** 0.5) / diagonal)


def _compute_synthetic_global_payload(
    descriptor_maps: Mapping[str, torch.Tensor],
    support_maps: Mapping[str, torch.Tensor],
) -> Dict[str, float | bool | str | int]:
    ir_hotspot = descriptor_maps["ir_hotspot"]
    ir_gradient = descriptor_maps["ir_gradient"]
    hotspot_mask = ir_hotspot > 0.0
    hotspot_values = ir_hotspot[hotspot_mask]

    pc_density = support_maps["pc_density"]
    valid_geometry = pc_density > 0.0
    pc_curvature = descriptor_maps["pc_curvature"]
    pc_roughness = descriptor_maps["pc_roughness"]
    pc_normal_deviation = descriptor_maps["pc_normal_deviation"]
    cross_agreement = descriptor_maps["cross_agreement"]
    cross_inconsistency = support_maps["cross_inconsistency"]
    cross_ir_support = support_maps["cross_ir_support"]
    cross_geo_support = support_maps["cross_geo_support"]

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        selected = values[mask]
        return float(selected.mean().item()) if selected.numel() else 0.0

    def masked_percentile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> float:
        selected = values[mask]
        if selected.numel() == 0:
            return 0.0
        return float(torch.quantile(selected.float(), quantile).item())

    return {
        "ir_hotspot_area_fraction": round(float(hotspot_mask.float().mean().item()), 6),
        "ir_max_gradient": round(float(ir_gradient.max().item()), 6),
        "ir_mean_hotspot_intensity": round(float(hotspot_values.mean().item()) if hotspot_values.numel() else 0.0, 6),
        "ir_hotspot_compactness": round(_mask_compactness(hotspot_mask.float()), 6),
        "pc_mean_curvature": round(masked_mean(pc_curvature, valid_geometry), 6),
        "pc_max_roughness": round(float(pc_roughness[valid_geometry].max().item()) if valid_geometry.any() else 0.0, 6),
        "pc_p95_normal_deviation": round(masked_percentile(pc_normal_deviation, valid_geometry, 0.95), 6),
        "cross_overlap_score": round(float(cross_agreement.mean().item()), 6),
        "cross_centroid_distance": round(
            _top_percent_centroid_distance(cross_ir_support, cross_geo_support, percentile=95.0),
            6,
        ),
        "cross_inconsistency_score": round(float(cross_inconsistency.mean().item()), 6),
        "cross_alignment_passed": True,
        "cross_alignment_message": "synthetic_validation_bundle",
        "cross_alignment_max_delta": 0,
    }


def _materialize_synthetic_validation_rows(
    calibration_rows: Sequence[ProcessedSampleRecord],
    *,
    output_root: Path,
) -> list[ProcessedSampleRecord]:
    synthetic_rows: list[ProcessedSampleRecord] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for row in calibration_rows:
        source_sample_dir = _resolve_sample_dir(row.sample_dir)
        rgb = _load_rgb_tensor(source_sample_dir / "rgb.png")
        descriptor_maps = _load_gray_stack(source_sample_dir, SPATIAL_DESCRIPTOR_ORDER)
        support_maps = _load_gray_stack(source_sample_dir, SUPPORT_DESCRIPTOR_ORDER)

        for recipe in SYNTHETIC_VALIDATION_RECIPES:
            sample_id = f"{row.sample_id}_{recipe}"
            sample_name = f"{row.category}__synthetic_validation__{recipe}__{sample_id}"
            bundle_dir = output_root / sample_name
            if _sample_is_complete(bundle_dir):
                existing_meta = _read_json(bundle_dir / "meta.json")
                synthetic_rows.append(
                    ProcessedSampleRecord(
                        category=str(existing_meta.get("category", row.category)),
                        split=str(existing_meta.get("split", "synthetic_validation")),
                        defect_label=str(existing_meta.get("defect_label", recipe)),
                        sample_id=str(existing_meta.get("sample_id", sample_id)),
                        sample_name=str(existing_meta.get("sample_name", sample_name)),
                        sample_dir=str(bundle_dir),
                        is_anomalous=True,
                        gt_mask_path="",
                        rgb_source_path=str(existing_meta.get("rgb_source_path", row.rgb_source_path)),
                        ir_source_path=str(existing_meta.get("ir_source_path", row.ir_source_path)),
                        pc_source_path=str(existing_meta.get("pc_source_path", row.pc_source_path)),
                    )
                )
                continue
            bundle_dir.mkdir(parents=True, exist_ok=True)

            mask = _synthetic_mask((int(rgb.shape[1]), int(rgb.shape[2])), sample_name=row.sample_name, recipe=recipe)
            synthetic_descriptors, synthetic_support = _apply_synthetic_recipe(
                recipe,
                descriptor_maps,
                support_maps,
                mask,
            )
            synthetic_rgb = _render_synthetic_rgb(rgb, mask, recipe)
            synthetic_global = _compute_synthetic_global_payload(synthetic_descriptors, synthetic_support)

            _save_rgb_tensor(synthetic_rgb, bundle_dir / "rgb.png")
            _save_gray_stack(bundle_dir, synthetic_descriptors)
            _save_gray_stack(bundle_dir, synthetic_support)
            (bundle_dir / "global.json").write_text(
                json.dumps(synthetic_global, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            meta_payload = {
                "category": row.category,
                "split": "synthetic_validation",
                "defect_label": recipe,
                "sample_id": sample_id,
                "sample_name": sample_name,
                "rgb_source_path": row.rgb_source_path,
                "ir_source_path": row.ir_source_path,
                "pc_source_path": row.pc_source_path,
                "gt_mask_path": None,
            }
            (bundle_dir / "meta.json").write_text(
                json.dumps(meta_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            synthetic_rows.append(
                ProcessedSampleRecord(
                    category=row.category,
                    split="synthetic_validation",
                    defect_label=recipe,
                    sample_id=sample_id,
                    sample_name=sample_name,
                    sample_dir=str(bundle_dir),
                    is_anomalous=True,
                    gt_mask_path="",
                    rgb_source_path=row.rgb_source_path,
                    ir_source_path=row.ir_source_path,
                    pc_source_path=row.pc_source_path,
                )
            )
    return synthetic_rows


def build_runtime_training_manifests(
    *,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    categories: Sequence[str] | None = None,
    manifests_root: Path | str | None = None,
    eval_subset: str = "official_test",
) -> Dict[str, object]:
    data_root = Path(data_root)
    resolved_processed_root = resolve_processed_root(data_root=data_root, processed_root=processed_root)
    samples_root = resolved_processed_root / "samples"
    selected_categories = _normalize_category_list(categories, samples_root=samples_root)
    rows = select_processed_sample_records(
        data_root=data_root,
        processed_root=resolved_processed_root,
        categories=selected_categories,
    )
    train_good_rows = [row for row in rows if row.split == "train" and not row.is_anomalous]
    official_test_rows = [row for row in rows if row.split == "test"]
    train_core_rows, calibration_rows = _split_train_core_and_calibration(train_good_rows)

    resolved_manifests_root = Path(manifests_root) if manifests_root is not None else data_root / "splits"
    resolved_manifests_root.mkdir(parents=True, exist_ok=True)
    synthetic_root = resolved_processed_root / "synthetic_validation_samples"
    synthetic_rows = _materialize_synthetic_validation_rows(calibration_rows, output_root=synthetic_root)
    selection_rows = list(calibration_rows) + list(synthetic_rows)

    train_core_csv = resolved_manifests_root / "train_core_good_manifest.csv"
    calibration_csv = resolved_manifests_root / "calibration_good_manifest.csv"
    synthetic_csv = resolved_manifests_root / "synthetic_validation_manifest.csv"
    official_test_csv = resolved_manifests_root / "official_test_manifest.csv"
    selection_csv = resolved_manifests_root / "selection_manifest.csv"
    train_csv = resolved_manifests_root / "train_manifest.csv"
    eval_csv = resolved_manifests_root / "eval_manifest.csv"

    _write_manifest_csv(train_core_csv, train_core_rows)
    _write_manifest_csv(calibration_csv, calibration_rows)
    _write_manifest_csv(synthetic_csv, synthetic_rows)
    _write_manifest_csv(official_test_csv, official_test_rows)
    _write_manifest_csv(selection_csv, selection_rows)
    _write_manifest_csv(train_csv, train_core_rows)
    if eval_subset == "selection":
        _write_manifest_csv(eval_csv, selection_rows)
    elif eval_subset == "synthetic_validation":
        _write_manifest_csv(eval_csv, synthetic_rows)
    elif eval_subset == "calibration_good":
        _write_manifest_csv(eval_csv, calibration_rows)
    else:
        _write_manifest_csv(eval_csv, official_test_rows)

    global_stats_json = resolved_processed_root / "manifests" / "global_feature_stats.json"
    summary = {
        "categories": selected_categories,
        "processed_root": str(resolved_processed_root),
        "train_core_good_manifest_csv": str(train_core_csv),
        "calibration_good_manifest_csv": str(calibration_csv),
        "synthetic_validation_manifest_csv": str(synthetic_csv),
        "official_test_manifest_csv": str(official_test_csv),
        "selection_manifest_csv": str(selection_csv),
        "train_manifest_csv": str(train_csv),
        "eval_manifest_csv": str(eval_csv),
        "eval_subset": eval_subset,
        "global_stats_json": str(global_stats_json),
        "train_core_records": len(train_core_rows),
        "calibration_records": len(calibration_rows),
        "synthetic_validation_records": len(synthetic_rows),
        "selection_records": len(selection_rows),
        "official_test_records": len(official_test_rows),
        "train_records": len(train_core_rows),
        "eval_records": len(selection_rows) if eval_subset == "selection" else len(official_test_rows) if eval_subset == "official_test" else len(synthetic_rows) if eval_subset == "synthetic_validation" else len(calibration_rows),
    }
    summary_json = resolved_manifests_root / "manifest_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "train_csv": train_csv,
        "eval_csv": eval_csv,
        "global_stats_json": global_stats_json,
        "summary_json": summary_json,
        "train_rows": train_core_rows,
        "train_core_rows": train_core_rows,
        "calibration_rows": calibration_rows,
        "synthetic_rows": synthetic_rows,
        "selection_rows": selection_rows,
        "eval_rows": selection_rows if eval_subset == "selection" else official_test_rows if eval_subset == "official_test" else synthetic_rows if eval_subset == "synthetic_validation" else calibration_rows,
        "official_test_rows": official_test_rows,
        "selected_categories": selected_categories,
        "train_core_csv": train_core_csv,
        "calibration_csv": calibration_csv,
        "synthetic_csv": synthetic_csv,
        "selection_csv": selection_csv,
        "official_test_csv": official_test_csv,
    }


def build_training_manifests(
    descriptor_index_csv: Path | str | None = None,
    *,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    categories: Sequence[str] | None = None,
) -> Dict[str, Path]:
    del descriptor_index_csv
    payload = build_runtime_training_manifests(
        data_root=data_root,
        processed_root=processed_root,
        categories=categories,
        manifests_root=Path(data_root) / "splits",
    )
    return {
        "train_csv": payload["train_csv"],  # type: ignore[return-value]
        "eval_csv": payload["eval_csv"],  # type: ignore[return-value]
        "train_core_csv": payload["train_core_csv"],  # type: ignore[return-value]
        "calibration_csv": payload["calibration_csv"],  # type: ignore[return-value]
        "synthetic_csv": payload["synthetic_csv"],  # type: ignore[return-value]
        "selection_csv": payload["selection_csv"],  # type: ignore[return-value]
        "official_test_csv": payload["official_test_csv"],  # type: ignore[return-value]
        "global_stats_json": payload["global_stats_json"],  # type: ignore[return-value]
        "summary_json": payload["summary_json"],  # type: ignore[return-value]
    }


def _write_manifest_csv(path: Path, rows: Sequence[ProcessedSampleRecord]) -> None:
    fieldnames = [
        "category",
        "split",
        "defect_label",
        "sample_id",
        "sample_name",
        "sample_dir",
        "is_anomalous",
        "gt_mask_path",
        "rgb_source_path",
        "ir_source_path",
        "pc_source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


class DescriptorConditioningDataset(Dataset[ModelSample]):
    def __init__(
        self,
        manifest_csv: Path | str,
        *,
        data_root: Path | str = ".",
        target_size: tuple[int, int] = (256, 256),
        global_stats_path: Path | str | None = None,
        object_crop_enabled: bool = False,
        object_crop_margin_ratio: float = 0.10,
        object_crop_mask_source: str = "rgb_then_pc_density",
        rgb_normalization_mode: str = "none",
        rgb_normalization_stats_path: Path | str | None = None,
        rgb_normalization_stats_by_category_path: Path | str | None = None,
        global_vector_normalization_stats_path: Path | str | None = None,
        category_vocabulary: Sequence[str] | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.global_means, self.global_stds = _load_global_stats(global_stats_path)
        self.crop_config = ObjectCropConfig(
            enabled=object_crop_enabled,
            margin_ratio=object_crop_margin_ratio,
            mask_source=object_crop_mask_source,
        )
        self.rgb_normalization_mode = rgb_normalization_mode
        self.rgb_normalization_stats = _load_rgb_normalization_stats(rgb_normalization_stats_path)
        self.rgb_normalization_stats_by_category = _load_rgb_normalization_stats_by_category(
            rgb_normalization_stats_by_category_path
        )
        self.global_vector_normalization_stats_by_category = _load_global_vector_normalization_stats_by_category(
            global_vector_normalization_stats_path
        )
        self.records = self._read_manifest_rows(self.manifest_csv)
        self.descriptor_channels = len(SPATIAL_DESCRIPTOR_ORDER)
        self.support_channels = len(SUPPORT_DESCRIPTOR_ORDER)
        self.conditioning_channels = self.descriptor_channels + self.support_channels
        self.global_dim = len(GLOBAL_FEATURE_NAMES)
        resolved_vocabulary = list(category_vocabulary or sorted({row["category"] for row in self.records}))
        self.category_vocabulary = tuple(resolved_vocabulary)
        self.category_to_index = {category: index for index, category in enumerate(self.category_vocabulary)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ModelSample:
        row = self.records[index]
        sample_dir = self.data_root / row["sample_dir"]

        rgb_raw = _load_rgb_tensor(sample_dir / "rgb.png")
        descriptor_raw = torch.cat(
            [_load_gray_tensor(sample_dir / f"{name}.png") for name in SPATIAL_DESCRIPTOR_ORDER],
            dim=0,
        )
        support_raw = torch.cat(
            [_load_gray_tensor(sample_dir / f"{name}.png") for name in SUPPORT_DESCRIPTOR_ORDER],
            dim=0,
        )
        pc_density = support_raw[
            SUPPORT_DESCRIPTOR_ORDER.index("pc_density") : SUPPORT_DESCRIPTOR_ORDER.index("pc_density") + 1
        ]
        crop_box = estimate_object_crop_box(rgb_raw, pc_density, config=self.crop_config)
        rgb = _resize_tensor(_crop_tensor(rgb_raw, crop_box), self.target_size)
        descriptor_maps = _resize_tensor(_crop_tensor(descriptor_raw, crop_box), self.target_size)
        support_maps = _resize_tensor(_crop_tensor(support_raw, crop_box), self.target_size)
        resolved_rgb_stats = None
        if self.rgb_normalization_mode != "none":
            resolved_rgb_stats = (
                (self.rgb_normalization_stats_by_category or {}).get(row["category"])
                if self.rgb_normalization_stats_by_category
                else self.rgb_normalization_stats
            )
        rgb = _normalize_rgb(rgb, resolved_rgb_stats)

        global_payload = _read_json(sample_dir / "global.json")
        raw_global_vector = torch.tensor(
            [float(global_payload.get(name, 0.0)) for name in GLOBAL_FEATURE_NAMES],
            dtype=torch.float32,
        )
        global_vector = _normalize_global_vector(
            raw_global_vector,
            (self.global_vector_normalization_stats_by_category or {}).get(row["category"])
            if self.global_vector_normalization_stats_by_category
            else None,
            fallback_means=self.global_means,
            fallback_stds=self.global_stds,
        )

        return ModelSample(
            x_rgb=rgb,
            conditioning=DescriptorConditioning(
                descriptor_maps=descriptor_maps,
                support_maps=support_maps,
                global_vector=global_vector,
                raw_global_vector=raw_global_vector,
            ),
            category=row["category"],
            category_index=self.category_to_index.get(row["category"], 0),
            split=row["split"],
            defect_label=row["defect_label"],
            sample_id=row["sample_id"],
            sample_name=row["sample_name"],
            is_anomalous=_bool_from_string(row["is_anomalous"]),
            sample_dir=str(sample_dir),
            rgb_path=str(sample_dir / "rgb.png"),
            gt_mask_path=row["gt_mask_path"],
            source_rgb_path=row["rgb_source_path"],
            source_ir_path=row["ir_source_path"],
            source_pointcloud_path=row["pc_source_path"],
            crop_box_xyxy=crop_box,
            crop_source_size_hw=(int(rgb_raw.shape[1]), int(rgb_raw.shape[2])),
            rgb_normalization_mode=self.rgb_normalization_mode,
        )

    def set_rgb_normalization_stats(self, stats: RGBNormalizationStats | None) -> None:
        self.rgb_normalization_stats = stats

    def set_rgb_normalization_stats_by_category(
        self,
        stats_by_category: Mapping[str, RGBNormalizationStats] | None,
    ) -> None:
        self.rgb_normalization_stats_by_category = dict(stats_by_category) if stats_by_category is not None else None

    def set_global_vector_normalization_stats_by_category(
        self,
        stats_by_category: Mapping[str, GlobalVectorNormalizationStats] | None,
    ) -> None:
        self.global_vector_normalization_stats_by_category = (
            dict(stats_by_category) if stats_by_category is not None else None
        )

    @staticmethod
    def _read_manifest_rows(path: Path) -> List[Dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]


def compute_masked_rgb_normalization_stats(
    dataset: DescriptorConditioningDataset,
    *,
    category: str,
) -> RGBNormalizationStats:
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    masked_pixels = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        object_mask = estimate_object_mask_from_sample(sample.x_rgb, sample.conditioning.support_maps, crop_config=dataset.crop_config)
        region = object_mask.squeeze(0) > 0
        values = sample.x_rgb[:, region] if region.any() else sample.x_rgb.view(3, -1)
        channel_sum += values.sum(dim=1, dtype=torch.float64)
        channel_sum_sq += (values.double() ** 2).sum(dim=1)
        masked_pixels += int(values.shape[1])

    if masked_pixels <= 0:
        mean = torch.zeros(3, dtype=torch.float64)
        std = torch.ones(3, dtype=torch.float64)
    else:
        mean = channel_sum / masked_pixels
        variance = (channel_sum_sq / masked_pixels) - mean.square()
        std = variance.clamp_min(1e-6).sqrt()

    return RGBNormalizationStats(
        category=category,
        mode=dataset.rgb_normalization_mode,
        mean=tuple(float(value) for value in mean.tolist()),
        std=tuple(float(value) for value in std.tolist()),
        masked_pixels=masked_pixels,
    )


def compute_masked_rgb_normalization_stats_by_category(
    dataset: DescriptorConditioningDataset,
) -> Dict[str, RGBNormalizationStats]:
    channel_sum: Dict[str, torch.Tensor] = {}
    channel_sum_sq: Dict[str, torch.Tensor] = {}
    masked_pixels: Dict[str, int] = {}

    for index in range(len(dataset)):
        sample = dataset[index]
        category = sample.category
        object_mask = estimate_object_mask_from_sample(
            sample.x_rgb,
            sample.conditioning.support_maps,
            crop_config=dataset.crop_config,
        )
        region = object_mask.squeeze(0) > 0
        values = sample.x_rgb[:, region] if region.any() else sample.x_rgb.view(3, -1)
        channel_sum.setdefault(category, torch.zeros(3, dtype=torch.float64))
        channel_sum_sq.setdefault(category, torch.zeros(3, dtype=torch.float64))
        masked_pixels.setdefault(category, 0)
        channel_sum[category] += values.sum(dim=1, dtype=torch.float64)
        channel_sum_sq[category] += (values.double() ** 2).sum(dim=1)
        masked_pixels[category] += int(values.shape[1])

    stats_by_category: Dict[str, RGBNormalizationStats] = {}
    for category in sorted(channel_sum):
        pixel_count = masked_pixels.get(category, 0)
        if pixel_count <= 0:
            mean = torch.zeros(3, dtype=torch.float64)
            std = torch.ones(3, dtype=torch.float64)
        else:
            mean = channel_sum[category] / pixel_count
            variance = (channel_sum_sq[category] / pixel_count) - mean.square()
            std = variance.clamp_min(1e-6).sqrt()
        stats_by_category[category] = RGBNormalizationStats(
            category=category,
            mode=dataset.rgb_normalization_mode,
            mean=tuple(float(value) for value in mean.tolist()),
            std=tuple(float(value) for value in std.tolist()),
            masked_pixels=pixel_count,
        )
    return stats_by_category


def save_rgb_normalization_stats(stats: RGBNormalizationStats, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_rgb_normalization_stats_by_category(
    stats_by_category: Mapping[str, RGBNormalizationStats],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "per_category",
        "categories": {
            category: stats.to_dict()
            for category, stats in sorted(stats_by_category.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def compute_global_vector_normalization_stats_by_category(
    rows: Sequence[ProcessedSampleRecord],
    *,
    data_root: Path | str = ".",
) -> Dict[str, GlobalVectorNormalizationStats]:
    data_root = Path(data_root)
    grouped: Dict[str, List[torch.Tensor]] = {}
    for row in rows:
        sample_dir = data_root / row.sample_dir
        global_payload = _read_json(sample_dir / "global.json")
        grouped.setdefault(row.category, []).append(
            torch.tensor(
                [float(global_payload.get(name, 0.0)) for name in GLOBAL_FEATURE_NAMES],
                dtype=torch.float64,
            )
        )
    stats_by_category: Dict[str, GlobalVectorNormalizationStats] = {}
    for category, vectors in sorted(grouped.items()):
        stacked = torch.stack(vectors, dim=0) if vectors else torch.zeros((1, len(GLOBAL_FEATURE_NAMES)), dtype=torch.float64)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False).clamp_min(1e-6)
        stats_by_category[category] = GlobalVectorNormalizationStats(
            category=category,
            feature_names=tuple(GLOBAL_FEATURE_NAMES),
            mean=tuple(float(value) for value in mean.tolist()),
            std=tuple(float(value) for value in std.tolist()),
            sample_count=int(stacked.shape[0]),
        )
    return stats_by_category


def save_global_vector_normalization_stats_by_category(
    stats_by_category: Mapping[str, GlobalVectorNormalizationStats],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "per_category",
        "categories": {
            category: stats.to_dict()
            for category, stats in sorted(stats_by_category.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def collate_model_samples(samples: Sequence[ModelSample]) -> Dict[str, object]:
    return {
        "x_rgb": torch.stack([sample.x_rgb for sample in samples], dim=0),
        "descriptor_maps": torch.stack([sample.conditioning.descriptor_maps for sample in samples], dim=0),
        "support_maps": torch.stack([sample.conditioning.support_maps for sample in samples], dim=0),
        "global_vectors": torch.stack([sample.conditioning.global_vector for sample in samples], dim=0),
        "raw_global_vectors": torch.stack([sample.conditioning.raw_global_vector for sample in samples], dim=0),
        "categories": [sample.category for sample in samples],
        "category_indices": torch.tensor([sample.category_index for sample in samples], dtype=torch.long),
        "defect_labels": [sample.defect_label for sample in samples],
        "sample_ids": [sample.sample_id for sample in samples],
        "sample_names": [sample.sample_name for sample in samples],
        "is_anomalous": [sample.is_anomalous for sample in samples],
        "sample_dirs": [sample.sample_dir for sample in samples],
        "rgb_paths": [sample.rgb_path for sample in samples],
        "gt_mask_paths": [sample.gt_mask_path for sample in samples],
        "source_rgb_paths": [sample.source_rgb_path for sample in samples],
        "source_ir_paths": [sample.source_ir_path for sample in samples],
        "source_pointcloud_paths": [sample.source_pointcloud_path for sample in samples],
        "crop_boxes_xyxy": [sample.crop_box_xyxy for sample in samples],
        "crop_source_sizes_hw": [sample.crop_source_size_hw for sample in samples],
        "rgb_normalization_modes": [sample.rgb_normalization_mode for sample in samples],
    }


def validate_model_sample(sample: ModelSample) -> List[str]:
    issues: List[str] = []
    if sample.x_rgb.shape[0] != 3:
        issues.append("rgb_channels_must_equal_3")
    if sample.conditioning.descriptor_maps.shape[0] != len(SPATIAL_DESCRIPTOR_ORDER):
        issues.append("descriptor_channel_count_mismatch")
    if sample.conditioning.support_maps.shape[0] != len(SUPPORT_DESCRIPTOR_ORDER):
        issues.append("support_channel_count_mismatch")
    if sample.conditioning.global_vector.shape[0] != len(GLOBAL_FEATURE_NAMES):
        issues.append("global_vector_dim_mismatch")
    if sample.conditioning.raw_global_vector.shape[0] != len(GLOBAL_FEATURE_NAMES):
        issues.append("raw_global_vector_dim_mismatch")
    if sample.category_index < 0:
        issues.append("category_index_negative")
    for name, tensor in {
        "x_rgb": sample.x_rgb,
        "descriptor_maps": sample.conditioning.descriptor_maps,
        "support_maps": sample.conditioning.support_maps,
        "global_vector": sample.conditioning.global_vector,
        "raw_global_vector": sample.conditioning.raw_global_vector,
    }.items():
        if not torch.isfinite(tensor).all():
            issues.append(f"{name}_contains_non_finite")
    if sample.rgb_normalization_mode == "none":
        if float(sample.x_rgb.min()) < 0.0 or float(sample.x_rgb.max()) > 1.0:
            issues.append("rgb_outside_unit_interval")
    if float(sample.conditioning.descriptor_maps.min()) < 0.0 or float(sample.conditioning.descriptor_maps.max()) > 1.0:
        issues.append("descriptor_maps_outside_unit_interval")
    if float(sample.conditioning.support_maps.min()) < 0.0 or float(sample.conditioning.support_maps.max()) > 1.0:
        issues.append("support_maps_outside_unit_interval")
    return issues
