import csv
import json
from pathlib import Path

from PIL import Image
import torch

from src.data_loader import (
    DescriptorConditioningDataset,
    build_training_manifests,
    collate_model_samples,
    validate_model_sample,
)
from src.preprocessing.dataset import build_dataset_index
from src.preprocessing.descriptor_dataset import build_descriptor_index
from src.training.train_diffusion import run_smoke_experiment


ASCII_STL = """solid test
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid test
"""


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (12, 12), value).save(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_descriptor_bundle(root: Path, row: dict[str, str], sample_offset: float) -> None:
    category = row["category"]
    split = row["split"]
    defect = row["defect_label"]
    sample_id = row["sample_id"]

    ir_dir = root / "processed" / "descriptors" / "infrared" / category / split / defect / sample_id
    pc_dir = root / "processed" / "descriptors" / "pointcloud" / category / split / defect / sample_id
    cm_dir = root / "processed" / "descriptors" / "crossmodal" / category / split / defect / sample_id
    for directory in [ir_dir, pc_dir, cm_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    ir_spatial = {
        "infrared_normalized": ir_dir / "normalized.png",
        "infrared_gradient_magnitude": ir_dir / "gradient_magnitude.png",
        "infrared_local_variance": ir_dir / "local_variance.png",
        "infrared_hotspot": ir_dir / "hotspot.png",
    }
    for index, (name, path) in enumerate(ir_spatial.items(), start=1):
        _write_gray(path, min(255, 25 * index))
    ir_global = {
        "hotspot_area_fraction": 0.1 + sample_offset,
        "maximum_thermal_gradient": 10.0 + sample_offset,
        "mean_hotspot_intensity": 20.0 + sample_offset,
        "hotspot_compactness": 0.3 + sample_offset,
        "gradient_mean": 4.0 + sample_offset,
        "mean_intensity": 100.0 + sample_offset,
        "std_intensity": 8.0 + sample_offset,
    }
    (ir_dir / "global.json").write_text(json.dumps(ir_global), encoding="utf-8")
    (ir_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "kind": "spatial",
                    "path": str(path),
                    "channels": 1,
                    "width": 12,
                    "height": 12,
                    "vector_length": 0,
                    "dtype": "uint8",
                }
                for name, path in ir_spatial.items()
            ]
            + [
                {
                    "name": "infrared_global",
                    "kind": "global",
                    "path": str(ir_dir / "global.json"),
                    "channels": 0,
                    "width": 0,
                    "height": 0,
                    "vector_length": 7,
                    "dtype": "json",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    pc_spatial = {
        "pointcloud_depth_topdown": pc_dir / "depth_topdown.png",
        "pointcloud_density_topdown": pc_dir / "density_topdown.png",
        "pointcloud_curvature": pc_dir / "curvature.png",
        "pointcloud_roughness": pc_dir / "roughness.png",
        "pointcloud_normal_deviation": pc_dir / "normal_deviation.png",
    }
    for index, (name, path) in enumerate(pc_spatial.items(), start=1):
        _write_gray(path, min(255, 30 * index))
    pc_global = {
        "mean_pointcloud_curvature": 1.0 + sample_offset,
        "maximum_pointcloud_roughness": 2.0 + sample_offset,
        "p95_normal_deviation": 3.0 + sample_offset,
        "coverage_ratio": 0.4 + sample_offset,
        "vertex_count": 1000.0 + sample_offset,
        "x_range": 4.0 + sample_offset,
        "y_range": 5.0 + sample_offset,
        "z_range": 6.0 + sample_offset,
    }
    (pc_dir / "global.json").write_text(json.dumps(pc_global), encoding="utf-8")
    (pc_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "kind": "spatial",
                    "path": str(path),
                    "channels": 1,
                    "width": 12,
                    "height": 12,
                    "vector_length": 0,
                    "dtype": "uint8",
                }
                for name, path in pc_spatial.items()
            ]
            + [
                {
                    "name": "pointcloud_global",
                    "kind": "global",
                    "path": str(pc_dir / "global.json"),
                    "channels": 0,
                    "width": 0,
                    "height": 0,
                    "vector_length": 8,
                    "dtype": "json",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    cm_spatial = {
        "crossmodal_infrared_support": cm_dir / "infrared_support.png",
        "crossmodal_geometric_support": cm_dir / "geometric_support.png",
        "crossmodal_agreement": cm_dir / "agreement.png",
        "crossmodal_inconsistency": cm_dir / "inconsistency.png",
    }
    for index, (name, path) in enumerate(cm_spatial.items(), start=1):
        _write_gray(path, min(255, 35 * index))
    cm_global = {
        "agreement_mean": 0.5 + sample_offset,
        "agreement_std": 0.2 + sample_offset,
        "support_mean": 0.7 + sample_offset,
        "ir_pc_overlap_score": 0.1 + sample_offset,
        "ir_pc_centroid_distance": 0.2 + sample_offset,
        "crossmodal_inconsistency_score": 0.3 + sample_offset,
    }
    (cm_dir / "global.json").write_text(json.dumps(cm_global), encoding="utf-8")
    (cm_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "kind": "spatial",
                    "path": str(path),
                    "channels": 1,
                    "width": 12,
                    "height": 12,
                    "vector_length": 0,
                    "dtype": "uint8",
                }
                for name, path in cm_spatial.items()
            ]
            + [
                {
                    "name": "crossmodal_global",
                    "kind": "global",
                    "path": str(cm_dir / "global.json"),
                    "channels": 0,
                    "width": 0,
                    "height": 0,
                    "vector_length": 6,
                    "dtype": "json",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_descriptor_ready_dataset(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw" / "MulSen_AD" / "capsule"
    for sample_id, color in [("0", (120, 30, 30)), ("1", (30, 120, 30))]:
        _write_rgb(raw_root / "RGB" / "train" / f"{sample_id}.png", color)
        _write_gray(raw_root / "Infrared" / "train" / f"{sample_id}.png", 180)
        _write_text(raw_root / "Pointcloud" / "train" / f"{sample_id}.stl", ASCII_STL)

    _write_rgb(raw_root / "RGB" / "test" / "good" / "0.png", (80, 80, 120))
    _write_gray(raw_root / "Infrared" / "test" / "good" / "0.png", 160)
    _write_text(raw_root / "Pointcloud" / "test" / "good" / "0.stl", ASCII_STL)

    build_dataset_index(raw_root=tmp_path / "raw" / "MulSen_AD", data_root=tmp_path)

    index_csv = tmp_path / "processed" / "manifests" / "dataset_index.csv"
    with index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for offset, row in enumerate(rows):
        _write_descriptor_bundle(tmp_path, row, sample_offset=offset * 0.1)

    build_descriptor_index(
        data_root=tmp_path,
        sample_index_csv=tmp_path / "processed" / "manifests" / "dataset_index.csv",
    )
    return tmp_path


def test_training_dataset_loader_returns_aligned_model_sample(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    manifests = build_training_manifests(
        descriptor_index_csv=root / "processed" / "manifests" / "descriptor_index.csv",
        data_root=root,
        categories=["capsule"],
    )

    dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        data_root=".",
        target_size=(8, 8),
        global_stats_path=manifests["global_stats_json"],
    )

    sample = dataset[0]
    issues = validate_model_sample(sample)
    assert issues == []
    assert isinstance(sample.x_rgb, torch.Tensor)
    assert sample.x_rgb.shape == (3, 8, 8)
    assert sample.conditioning.descriptor_maps.shape == (13, 8, 8)
    assert sample.conditioning.global_vector.shape == (21,)
    assert 0.0 <= float(sample.x_rgb.min()) <= float(sample.x_rgb.max()) <= 1.0
    assert 0.0 <= float(sample.conditioning.descriptor_maps.min()) <= float(sample.conditioning.descriptor_maps.max()) <= 1.0

    batch = collate_model_samples([dataset[0], dataset[1]])
    assert isinstance(batch["x_rgb"], torch.Tensor)
    assert batch["x_rgb"].shape == (2, 3, 8, 8)
    assert batch["descriptor_maps"].shape == (2, 13, 8, 8)
    assert batch["global_vectors"].shape == (2, 21)


def test_capsule_smoke_experiment_runs_on_descriptor_ready_data(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    result = run_smoke_experiment(
        data_root=root,
        descriptor_index_csv=root / "processed" / "manifests" / "descriptor_index.csv",
        category="capsule",
        batch_size=1,
        max_batches=2,
        target_size=(8, 8),
    )

    summary = result["summary"]
    assert summary["category"] == "capsule"
    assert summary["batches_run"] == 2
    assert summary["train_records"] == 2
    assert result["summary_path"].exists()
    assert summary["mean_loss"] >= 0.0
