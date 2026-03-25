import json
from pathlib import Path
import csv

from PIL import Image
import torch

from src.data_loader import (
    DescriptorConditioningDataset,
    GLOBAL_FEATURE_NAMES,
    SPATIAL_DESCRIPTOR_ORDER,
    SUPPORT_DESCRIPTOR_ORDER,
    build_runtime_training_manifests,
    build_training_manifests,
    collate_model_samples,
    compute_masked_rgb_normalization_stats,
    validate_model_sample,
)
from src.preprocessing.dataset import build_dataset_index
from src.preprocessing.descriptor_pipeline import run_descriptor_pipeline
from src.training.train_diffusion import build_diffusion_dataloaders, run_smoke_experiment


ASCII_STL = """solid test
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 1 0 0
vertex 1 1 0
vertex 0 1 0
endloop
endfacet
endsolid test
"""


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), value).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (32, 32), 0)
    for x in range(10, 22):
        for y in range(10, 22):
            image.putpixel((x, y), 255)
    image.save(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_descriptor_ready_dataset(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw" / "MulSen_AD" / "capsule"

    for sample_id, color, ir_value in [
        ("0", (120, 30, 30), 120),
        ("1", (30, 120, 30), 140),
    ]:
        _write_rgb(raw_root / "RGB" / "train" / f"{sample_id}.png", color)
        _write_gray(raw_root / "Infrared" / "train" / f"{sample_id}.png", ir_value)
        _write_text(raw_root / "Pointcloud" / "train" / f"{sample_id}.stl", ASCII_STL)

    _write_rgb(raw_root / "RGB" / "test" / "good" / "0.png", (80, 80, 120))
    _write_gray(raw_root / "Infrared" / "test" / "good" / "0.png", 125)
    _write_text(raw_root / "Pointcloud" / "test" / "good" / "0.stl", ASCII_STL)

    _write_rgb(raw_root / "RGB" / "test" / "crack" / "0.png", (150, 40, 40))
    _write_gray(raw_root / "Infrared" / "test" / "crack" / "0.png", 210)
    _write_text(raw_root / "Pointcloud" / "test" / "crack" / "0.stl", ASCII_STL)
    _write_mask(raw_root / "RGB" / "GT" / "crack" / "0.png")

    build_dataset_index(raw_root=tmp_path / "raw" / "MulSen_AD", data_root=tmp_path)
    run_descriptor_pipeline(
        data_root=tmp_path,
        index_csv=tmp_path / "processed" / "manifests" / "dataset_index.csv",
        skip_existing=False,
        continue_on_error=False,
    )
    return tmp_path


def _write_processed_sample(
    root: Path,
    *,
    sample_name: str,
    split: str,
    defect_label: str,
    rgb_color: tuple[int, int, int],
) -> None:
    sample_dir = root / "processed" / "samples" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb = Image.new("RGB", (64, 64), (0, 0, 0))
    for x in range(20, 44):
        for y in range(18, 46):
            rgb.putpixel((x, y), rgb_color)
    rgb.save(sample_dir / "rgb.png")

    def _save_rect_gray(path: Path, value: int) -> None:
        image = Image.new("L", (64, 64), 0)
        for x in range(20, 44):
            for y in range(18, 46):
                image.putpixel((x, y), value)
        image.save(path)

    for name in SPATIAL_DESCRIPTOR_ORDER:
        _save_rect_gray(sample_dir / f"{name}.png", 180 if name != "cross_agreement" else 220)
    for name in SUPPORT_DESCRIPTOR_ORDER:
        _save_rect_gray(sample_dir / f"{name}.png", 200 if name == "pc_density" else 160)

    (sample_dir / "global.json").write_text(
        json.dumps({name: 0.0 for name in GLOBAL_FEATURE_NAMES}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (sample_dir / "meta.json").write_text(
        json.dumps(
            {
                "category": "capsule",
                "defect_label": defect_label,
                "gt_mask_path": "" if defect_label == "good" else str(sample_dir / "gt_mask.png"),
                "ir_source_path": f"data/raw/MulSen_AD/capsule/Infrared/{split}/{defect_label}/0.png",
                "pc_source_path": f"data/raw/MulSen_AD/capsule/Pointcloud/{split}/{defect_label}/0.stl",
                "rgb_source_path": f"data/raw/MulSen_AD/capsule/RGB/{split}/{defect_label}/0.png",
                "sample_id": sample_name.rsplit("__", 1)[-1],
                "sample_name": sample_name,
                "split": split,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if defect_label != "good":
        _write_mask(sample_dir / "gt_mask.png")


def _build_processed_manifest_ready_dataset(tmp_path: Path) -> Path:
    root = tmp_path
    manifests_root = root / "processed" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / "global_feature_stats.json").write_text(
        json.dumps(
            {
                "feature_names": GLOBAL_FEATURE_NAMES,
                "means": {name: 0.0 for name in GLOBAL_FEATURE_NAMES},
                "stds": {name: 1.0 for name in GLOBAL_FEATURE_NAMES},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_processed_sample(root, sample_name="capsule__train__good__000", split="train", defect_label="good", rgb_color=(204, 26, 26))
    _write_processed_sample(root, sample_name="capsule__train__good__001", split="train", defect_label="good", rgb_color=(51, 179, 26))
    _write_processed_sample(root, sample_name="capsule__test__good__000", split="test", defect_label="good", rgb_color=(128, 128, 26))
    _write_processed_sample(root, sample_name="capsule__test__crack__000", split="test", defect_label="crack", rgb_color=(26, 26, 230))
    return root


def _build_multi_category_processed_manifest_ready_dataset(tmp_path: Path) -> Path:
    root = tmp_path
    manifests_root = root / "processed" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / "global_feature_stats.json").write_text(
        json.dumps(
            {
                "feature_names": GLOBAL_FEATURE_NAMES,
                "means": {name: 0.0 for name in GLOBAL_FEATURE_NAMES},
                "stds": {name: 1.0 for name in GLOBAL_FEATURE_NAMES},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for category, base_color in [("capsule", (204, 26, 26)), ("screw", (26, 153, 204))]:
        _write_processed_sample(root, sample_name=f"{category}__train__good__000", split="train", defect_label="good", rgb_color=base_color)
        _write_processed_sample(root, sample_name=f"{category}__train__good__001", split="train", defect_label="good", rgb_color=tuple(min(channel + 20, 255) for channel in base_color))
        _write_processed_sample(root, sample_name=f"{category}__test__good__000", split="test", defect_label="good", rgb_color=tuple(min(channel + 40, 255) for channel in base_color))
        _write_processed_sample(root, sample_name=f"{category}__test__crack__000", split="test", defect_label="crack", rgb_color=(230, 26, 26))
        for sample_name in [f"{category}__train__good__000", f"{category}__train__good__001", f"{category}__test__good__000", f"{category}__test__crack__000"]:
            sample_dir = root / "processed" / "samples" / sample_name
            meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
            meta["category"] = category
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return root


def test_training_dataset_loader_returns_aligned_model_sample(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    manifests = build_training_manifests(data_root=root, categories=["capsule"])

    dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        target_size=(16, 16),
        global_stats_path=manifests["global_stats_json"],
    )

    sample = dataset[0]
    issues = validate_model_sample(sample)
    assert issues == []
    assert isinstance(sample.x_rgb, torch.Tensor)
    assert sample.x_rgb.shape == (3, 16, 16)
    assert sample.conditioning.descriptor_maps.shape == (7, 16, 16)
    assert sample.conditioning.support_maps.shape == (6, 16, 16)
    assert sample.conditioning.global_vector.shape == (10,)
    assert 0.0 <= float(sample.x_rgb.min()) <= float(sample.x_rgb.max()) <= 1.0
    assert 0.0 <= float(sample.conditioning.descriptor_maps.min()) <= float(sample.conditioning.descriptor_maps.max()) <= 1.0
    assert 0.0 <= float(sample.conditioning.support_maps.min()) <= float(sample.conditioning.support_maps.max()) <= 1.0

    batch = collate_model_samples([sample, dataset[1]])
    assert batch["descriptor_maps"].shape == (2, 7, 16, 16)
    assert batch["support_maps"].shape == (2, 6, 16, 16)

    with manifests["train_csv"].open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    with manifests["eval_csv"].open(newline="", encoding="utf-8") as handle:
        eval_rows = list(csv.DictReader(handle))
    assert train_rows
    assert all(row["split"] == "train" and row["defect_label"] == "good" and row["is_anomalous"] == "False" for row in train_rows)
    assert eval_rows
    assert {row["defect_label"] for row in eval_rows} == {"good", "crack"}


def test_capsule_smoke_experiment_runs_on_descriptor_ready_data(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    result = run_smoke_experiment(
        data_root=root,
        category="capsule",
        batch_size=1,
        max_batches=1,
        target_size=(16, 16),
        device="cpu",
    )

    summary = result["summary"]
    assert summary["train_records"] == 2
    assert summary["eval_records"] == 2
    assert summary["batches_run"] == 1
    assert Path(result["summary_path"]).exists()


def test_loader_object_crop_is_stable_and_improves_object_coverage(tmp_path: Path) -> None:
    root = _build_processed_manifest_ready_dataset(tmp_path)
    manifests = build_training_manifests(data_root=root, categories=["capsule"])

    uncropped = DescriptorConditioningDataset(
        manifests["train_csv"],
        target_size=(32, 32),
        global_stats_path=manifests["global_stats_json"],
    )
    cropped = DescriptorConditioningDataset(
        manifests["train_csv"],
        target_size=(32, 32),
        global_stats_path=manifests["global_stats_json"],
        object_crop_enabled=True,
        object_crop_margin_ratio=0.1,
    )

    sample_uncropped = uncropped[0]
    sample_cropped_a = cropped[0]
    sample_cropped_b = cropped[0]
    assert sample_cropped_a.crop_box_xyxy is not None
    assert sample_cropped_a.crop_box_xyxy == sample_cropped_b.crop_box_xyxy
    assert sample_cropped_a.conditioning.descriptor_maps.shape == (7, 32, 32)
    assert sample_cropped_a.conditioning.support_maps.shape == (6, 32, 32)
    uncropped_object_fraction = float((sample_uncropped.x_rgb.mean(dim=0) > 0.01).float().mean().item())
    cropped_object_fraction = float((sample_cropped_a.x_rgb.mean(dim=0) > 0.01).float().mean().item())
    assert cropped_object_fraction > uncropped_object_fraction


def test_masked_rgb_normalization_stats_use_train_good_only(tmp_path: Path) -> None:
    root = _build_processed_manifest_ready_dataset(tmp_path)
    loaders = build_diffusion_dataloaders(
        data_root=root,
        category="capsule",
        batch_size=1,
        target_size=(32, 32),
        object_crop_enabled=True,
        rgb_normalization_mode="per_category_masked",
    )
    stats = loaders["rgb_normalization_stats"]
    assert stats is not None
    assert stats.mode == "per_category_masked"
    assert stats.masked_pixels > 0
    assert all(torch.isfinite(torch.tensor(stats.mean)))
    assert all(torch.tensor(stats.std) > 0)
    assert abs(stats.mean[0] - 0.5) < 0.05
    assert abs(stats.mean[1] - 0.40196) < 0.06
    assert abs(stats.mean[2] - 0.10196) < 0.03

    sample = loaders["train_dataset"][0]
    assert torch.isfinite(sample.x_rgb).all()
    raw_dataset = DescriptorConditioningDataset(
        loaders["manifests"]["train_csv"],
        target_size=(32, 32),
        global_stats_path=loaders["manifests"]["global_stats_json"],
        object_crop_enabled=True,
        rgb_normalization_mode="per_category_masked",
    )
    recomputed = compute_masked_rgb_normalization_stats(raw_dataset, category="capsule")
    assert recomputed.masked_pixels == stats.masked_pixels


def test_runtime_manifests_and_joint_loader_support_multiple_categories(tmp_path: Path) -> None:
    root = _build_multi_category_processed_manifest_ready_dataset(tmp_path)
    manifests = build_runtime_training_manifests(
        data_root=root,
        categories=["capsule", "screw"],
        manifests_root=root / "runtime_manifests",
    )

    assert manifests["selected_categories"] == ["capsule", "screw"]
    assert len(manifests["train_rows"]) == 4
    assert len(manifests["eval_rows"]) == 4

    loaders = build_diffusion_dataloaders(
        data_root=root,
        categories=["capsule", "screw"],
        batch_size=2,
        target_size=(32, 32),
        object_crop_enabled=True,
        rgb_normalization_mode="per_category_masked",
        manifests_root=root / "runtime_manifests",
        sampler_mode="balanced_by_category",
    )

    assert loaders["selected_categories"] == ["capsule", "screw"]
    assert loaders["category_vocabulary"] == ["capsule", "screw"]
    assert set(loaders["train_category_counts"].keys()) == {"capsule", "screw"}
    assert set(loaders["rgb_normalization_stats_by_category"].keys()) == {"capsule", "screw"}
    assert set(loaders["global_vector_normalization_stats_by_category"].keys()) == {"capsule", "screw"}

    sample = loaders["train_dataset"][0]
    assert sample.category in {"capsule", "screw"}
    assert sample.category_index in {0, 1}

    batch = next(iter(loaders["train_loader"]))
    assert "category_indices" in batch
    assert tuple(batch["category_indices"].shape) == (2,)
