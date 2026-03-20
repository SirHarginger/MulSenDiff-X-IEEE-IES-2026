from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime environments
    raise ModuleNotFoundError(
        "torch and torchvision are required for src.data_loader. Activate the project venv before running training code."
    ) from exc

from PIL import Image

from src.preprocessing.descriptor_dataset import DescriptorBundleAccessor, DescriptorIndexRecord, iter_descriptor_index


SPATIAL_DESCRIPTOR_ORDER = (
    "infrared_normalized",
    "infrared_gradient_magnitude",
    "infrared_local_variance",
    "infrared_hotspot",
    "pointcloud_depth_topdown",
    "pointcloud_density_topdown",
    "pointcloud_curvature",
    "pointcloud_roughness",
    "pointcloud_normal_deviation",
    "crossmodal_infrared_support",
    "crossmodal_geometric_support",
    "crossmodal_agreement",
    "crossmodal_inconsistency",
)

GLOBAL_DESCRIPTOR_ORDER = (
    ("infrared", "hotspot_area_fraction"),
    ("infrared", "maximum_thermal_gradient"),
    ("infrared", "mean_hotspot_intensity"),
    ("infrared", "hotspot_compactness"),
    ("infrared", "gradient_mean"),
    ("infrared", "mean_intensity"),
    ("infrared", "std_intensity"),
    ("pointcloud", "mean_pointcloud_curvature"),
    ("pointcloud", "maximum_pointcloud_roughness"),
    ("pointcloud", "p95_normal_deviation"),
    ("pointcloud", "coverage_ratio"),
    ("pointcloud", "vertex_count"),
    ("pointcloud", "x_range"),
    ("pointcloud", "y_range"),
    ("pointcloud", "z_range"),
    ("crossmodal", "agreement_mean"),
    ("crossmodal", "agreement_std"),
    ("crossmodal", "support_mean"),
    ("crossmodal", "ir_pc_overlap_score"),
    ("crossmodal", "ir_pc_centroid_distance"),
    ("crossmodal", "crossmodal_inconsistency_score"),
)

GLOBAL_FEATURE_NAMES = tuple(name for _, name in GLOBAL_DESCRIPTOR_ORDER)

_BINARY_LIKE_ARTIFACTS = {
    "infrared_hotspot",
    "crossmodal_infrared_support",
    "crossmodal_agreement",
    "crossmodal_inconsistency",
}


@dataclass(frozen=True)
class GlobalNormalizationStats:
    feature_names: Sequence[str]
    means: Dict[str, float]
    stds: Dict[str, float]


@dataclass(frozen=True)
class SampleValidationIssue:
    field: str
    message: str


@dataclass(frozen=True)
class ConditioningPackage:
    descriptor_maps: torch.Tensor
    global_vector: torch.Tensor
    spatial_names: Sequence[str]
    global_names: Sequence[str]


@dataclass(frozen=True)
class ModelSample:
    x_rgb: torch.Tensor
    conditioning: ConditioningPackage
    category: str
    split: str
    defect_label: str
    sample_id: str
    is_anomalous: bool
    rgb_path: str
    rgb_gt_path: str
    ir_gt_path: str
    pointcloud_gt_path: str


def build_training_manifests(
    *,
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    data_root: Path | str = "data",
    categories: Sequence[str] | None = None,
) -> Dict[str, Path]:
    data_root = Path(data_root)
    descriptor_index_csv = Path(descriptor_index_csv)
    categories_set = {item for item in (categories or []) if item}

    rows = [
        row
        for row in iter_descriptor_index(descriptor_index_csv)
        if row.descriptor_ready and (not categories_set or row.category in categories_set)
    ]
    train_rows = [row for row in rows if row.split == "train" and not row.is_anomalous]
    eval_rows = [row for row in rows if row.split == "test"]

    splits_root = data_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    train_csv = splits_root / "train_manifest.csv"
    eval_csv = splits_root / "eval_manifest.csv"
    summary_json = splits_root / "manifest_summary.json"
    stats_json = data_root / "processed" / "manifests" / "global_feature_stats.json"

    _write_descriptor_rows(train_rows, train_csv)
    _write_descriptor_rows(eval_rows, eval_csv)

    stats = fit_global_normalization_stats(train_rows, data_root=".")
    write_global_normalization_stats(stats, stats_json)

    summary = {
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "categories": sorted(categories_set) if categories_set else "all",
        "train_manifest_csv": str(train_csv),
        "eval_manifest_csv": str(eval_csv),
        "global_stats_json": str(stats_json),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "train_csv": train_csv,
        "eval_csv": eval_csv,
        "summary_json": summary_json,
        "global_stats_json": stats_json,
    }


def fit_global_normalization_stats(
    rows: Sequence[DescriptorIndexRecord],
    *,
    data_root: Path | str = ".",
) -> GlobalNormalizationStats:
    accessor = DescriptorBundleAccessor(data_root)
    measurements: Dict[str, List[float]] = {name: [] for name in GLOBAL_FEATURE_NAMES}

    for row in rows:
        vector = load_global_feature_vector(row, accessor)
        for name, value in zip(GLOBAL_FEATURE_NAMES, vector.tolist()):
            measurements[name].append(float(value))

    means = {}
    stds = {}
    for name, values in measurements.items():
        array = torch.tensor(values or [0.0], dtype=torch.float32)
        means[name] = float(array.mean().item())
        std = float(array.std(unbiased=False).item())
        stds[name] = std if std > 1e-6 else 1.0

    return GlobalNormalizationStats(feature_names=GLOBAL_FEATURE_NAMES, means=means, stds=stds)


def write_global_normalization_stats(stats: GlobalNormalizationStats, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": list(stats.feature_names),
        "means": stats.means,
        "stds": stats.stds,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_global_normalization_stats(path: Path | str) -> GlobalNormalizationStats:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GlobalNormalizationStats(
        feature_names=tuple(payload["feature_names"]),
        means={key: float(value) for key, value in payload["means"].items()},
        stds={key: float(value) for key, value in payload["stds"].items()},
    )


class DescriptorConditioningDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path | str,
        *,
        data_root: Path | str = ".",
        target_size: tuple[int, int] = (256, 256),
        global_stats_path: Path | str | None = None,
        strict: bool = True,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.strict = strict
        self.accessor = DescriptorBundleAccessor(self.data_root)
        self.rows = list(iter_descriptor_index(self.manifest_csv))
        self.global_stats = load_global_normalization_stats(global_stats_path) if global_stats_path else None

    @property
    def descriptor_channels(self) -> int:
        return len(SPATIAL_DESCRIPTOR_ORDER)

    @property
    def global_dim(self) -> int:
        return len(GLOBAL_FEATURE_NAMES)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ModelSample:
        row = self.rows[index]
        sample = load_model_sample(
            row,
            data_root=self.data_root,
            target_size=self.target_size,
            global_stats=self.global_stats,
        )
        issues = validate_model_sample(sample)
        if self.strict and issues:
            rendered = ", ".join(f"{issue.field}:{issue.message}" for issue in issues)
            raise ValueError(f"Model sample validation failed for {row.category}/{row.sample_id}: {rendered}")
        return sample


def load_model_sample(
    row: DescriptorIndexRecord,
    *,
    data_root: Path | str = ".",
    target_size: tuple[int, int] = (256, 256),
    global_stats: GlobalNormalizationStats | None = None,
) -> ModelSample:
    data_root = Path(data_root)
    accessor = DescriptorBundleAccessor(data_root)

    x_rgb = _load_rgb_tensor(accessor.resolve(row.rgb_path), target_size=target_size)
    descriptor_maps = _load_spatial_descriptor_stack(row, accessor, target_size=target_size)
    global_vector = load_global_feature_vector(row, accessor)
    if global_stats is not None:
        global_vector = normalize_global_vector(global_vector, global_stats)

    return ModelSample(
        x_rgb=x_rgb,
        conditioning=ConditioningPackage(
            descriptor_maps=descriptor_maps,
            global_vector=global_vector,
            spatial_names=SPATIAL_DESCRIPTOR_ORDER,
            global_names=GLOBAL_FEATURE_NAMES,
        ),
        category=row.category,
        split=row.split,
        defect_label=row.defect_label,
        sample_id=row.sample_id,
        is_anomalous=row.is_anomalous,
        rgb_path=row.rgb_path,
        rgb_gt_path=row.rgb_gt_path,
        ir_gt_path=row.ir_gt_path,
        pointcloud_gt_path=row.pointcloud_gt_path,
    )


def load_global_feature_vector(
    row: DescriptorIndexRecord,
    accessor: DescriptorBundleAccessor,
) -> torch.Tensor:
    payloads = {
        "infrared": accessor.load_global_summary(row.infrared_manifest_path),
        "pointcloud": accessor.load_global_summary(row.pointcloud_manifest_path),
        "crossmodal": accessor.load_global_summary(row.crossmodal_manifest_path),
    }
    values = []
    missing = []
    for source, key in GLOBAL_DESCRIPTOR_ORDER:
        if key not in payloads[source]:
            missing.append(f"{source}.{key}")
            values.append(0.0)
        else:
            values.append(float(payloads[source][key]))
    if missing:
        raise KeyError(f"Missing global descriptor features: {', '.join(missing)}")
    return torch.tensor(values, dtype=torch.float32)


def normalize_global_vector(vector: torch.Tensor, stats: GlobalNormalizationStats) -> torch.Tensor:
    means = torch.tensor([stats.means[name] for name in stats.feature_names], dtype=torch.float32)
    stds = torch.tensor([stats.stds[name] for name in stats.feature_names], dtype=torch.float32)
    return (vector.float() - means) / stds


def validate_model_sample(sample: ModelSample) -> List[SampleValidationIssue]:
    issues: List[SampleValidationIssue] = []
    rgb = sample.x_rgb
    maps = sample.conditioning.descriptor_maps
    global_vector = sample.conditioning.global_vector

    if rgb.ndim != 3 or rgb.shape[0] != 3:
        issues.append(SampleValidationIssue("x_rgb", f"expected_channels_first_rgb_got_{tuple(rgb.shape)}"))
    if maps.ndim != 3 or maps.shape[0] != len(SPATIAL_DESCRIPTOR_ORDER):
        issues.append(SampleValidationIssue("M", f"unexpected_descriptor_shape_{tuple(maps.shape)}"))
    if global_vector.ndim != 1 or global_vector.shape[0] != len(GLOBAL_FEATURE_NAMES):
        issues.append(SampleValidationIssue("g", f"unexpected_global_vector_length_{tuple(global_vector.shape)}"))

    if rgb.ndim == 3 and maps.ndim == 3 and rgb.shape[1:] != maps.shape[1:]:
        issues.append(SampleValidationIssue("channel_alignment", "rgb_and_descriptor_spatial_shapes_do_not_match"))

    if not torch.isfinite(rgb).all().item():
        issues.append(SampleValidationIssue("x_rgb", "contains_non_finite_values"))
    if not torch.isfinite(maps).all().item():
        issues.append(SampleValidationIssue("M", "contains_non_finite_values"))
    if not torch.isfinite(global_vector).all().item():
        issues.append(SampleValidationIssue("g", "contains_non_finite_values"))

    if rgb.numel() and (rgb.min().item() < -1e-6 or rgb.max().item() > 1.0 + 1e-6):
        issues.append(SampleValidationIssue("x_rgb", "rgb_out_of_range"))
    if maps.numel() and (maps.min().item() < -1e-6 or maps.max().item() > 1.0 + 1e-6):
        issues.append(SampleValidationIssue("M", "descriptor_maps_out_of_range"))

    return issues


def collate_model_samples(samples: Sequence[ModelSample]) -> Dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    return {
        "x_rgb": torch.stack([sample.x_rgb for sample in samples], dim=0),
        "descriptor_maps": torch.stack([sample.conditioning.descriptor_maps for sample in samples], dim=0),
        "global_vectors": torch.stack([sample.conditioning.global_vector for sample in samples], dim=0),
        "categories": [sample.category for sample in samples],
        "defect_labels": [sample.defect_label for sample in samples],
        "sample_ids": [sample.sample_id for sample in samples],
        "is_anomalous": [sample.is_anomalous for sample in samples],
        "rgb_paths": [sample.rgb_path for sample in samples],
        "rgb_gt_paths": [sample.rgb_gt_path for sample in samples],
        "ir_gt_paths": [sample.ir_gt_path for sample in samples],
        "pointcloud_gt_paths": [sample.pointcloud_gt_path for sample in samples],
    }


def _write_descriptor_rows(rows: Sequence[DescriptorIndexRecord], csv_path: Path) -> None:
    payload = [asdict(row) for row in rows]
    fieldnames = list(payload[0].keys()) if payload else list(DescriptorIndexRecord.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def _load_rgb_tensor(path: Path, *, target_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, list(target_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.pil_to_tensor(image).float() / 255.0


def _load_spatial_descriptor_stack(
    row: DescriptorIndexRecord,
    accessor: DescriptorBundleAccessor,
    *,
    target_size: tuple[int, int],
) -> torch.Tensor:
    manifests = [
        accessor.load_manifest(row.infrared_manifest_path),
        accessor.load_manifest(row.pointcloud_manifest_path),
        accessor.load_manifest(row.crossmodal_manifest_path),
    ]

    artifact_paths: Dict[str, str] = {}
    for manifest in manifests:
        for artifact in manifest:
            if artifact["kind"] == "spatial":
                artifact_paths[artifact["name"]] = artifact["path"]

    channels = []
    missing = [name for name in SPATIAL_DESCRIPTOR_ORDER if name not in artifact_paths]
    if missing:
        raise KeyError(f"Missing spatial descriptor artifacts: {', '.join(missing)}")

    for name in SPATIAL_DESCRIPTOR_ORDER:
        image = Image.open(accessor.resolve(artifact_paths[name])).convert("L")
        interpolation = InterpolationMode.NEAREST if name in _BINARY_LIKE_ARTIFACTS else InterpolationMode.BILINEAR
        image = TF.resize(image, list(target_size), interpolation=interpolation, antialias=True)
        channel = TF.pil_to_tensor(image).float() / 255.0
        channels.append(channel)
    return torch.cat(channels, dim=0)
