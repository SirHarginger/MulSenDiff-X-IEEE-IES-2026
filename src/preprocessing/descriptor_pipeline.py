from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from PIL import Image

from src.preprocessing.category_stats import build_category_stats
from src.preprocessing.crossmodal_descriptors import (
    generate_crossmodal_maps,
    save_grayscale_png as save_crossmodal_png,
    save_preview_png as save_crossmodal_preview_png,
)
from src.preprocessing.dataset import SampleRecord, iter_index
from src.preprocessing.descriptor_audit import audit_descriptor_root, write_descriptor_audit_reports
from src.preprocessing.descriptor_validation import validate_descriptor_root, write_validation_reports
from src.preprocessing.ir_descriptors import (
    generate_ir_sample_descriptors,
    load_ir_category_stats,
    save_grayscale_png as save_ir_png,
)
from src.preprocessing.pointcloud_projector import (
    generate_pointcloud_sample_descriptors,
    load_pointcloud_category_stats,
)


@dataclass(frozen=True)
class DescriptorPipelineFailure:
    category: str
    sample_id: str
    split: str
    defect_label: str
    stage: str
    message: str


def select_records(
    records: Iterable[SampleRecord],
    *,
    categories: Optional[Set[str]] = None,
    splits: Optional[Set[str]] = None,
    defect_labels: Optional[Set[str]] = None,
    include_anomalous: bool = True,
    include_normal: bool = True,
    limit: int = 0,
) -> List[SampleRecord]:
    selected: List[SampleRecord] = []
    for record in records:
        if categories and record.category not in categories:
            continue
        if splits and record.split not in splits:
            continue
        if defect_labels and record.defect_label not in defect_labels:
            continue
        if record.is_anomalous and not include_anomalous:
            continue
        if not record.is_anomalous and not include_normal:
            continue
        selected.append(record)
        if limit and len(selected) >= limit:
            break
    return selected


def run_descriptor_pipeline(
    *,
    data_root: Path | str = "data",
    index_csv: Path | str = "data/processed/manifests/dataset_index.csv",
    categories: Optional[Set[str]] = None,
    splits: Optional[Set[str]] = None,
    defect_labels: Optional[Set[str]] = None,
    include_anomalous: bool = True,
    include_normal: bool = True,
    limit: int = 0,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    target_size: tuple[int, int] = (256, 256),
    log_progress: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    data_root = Path(data_root)
    index_csv = Path(index_csv)
    records = select_records(
        iter_index(index_csv),
        categories=categories,
        splits=splits,
        defect_labels=defect_labels,
        include_anomalous=include_anomalous,
        include_normal=include_normal,
        limit=limit,
    )

    selected_categories = sorted({record.category for record in records})
    _log(
        log_progress,
        f"descriptor_pipeline: selected {len(records)} records across {len(selected_categories)} categories",
    )
    raw_root = data_root / "raw" / "MulSen_AD"
    _log(log_progress, "descriptor_pipeline: building category stats")
    category_stats_result = build_category_stats(
        raw_root=raw_root,
        data_root=data_root,
        categories=selected_categories,
        target_size=target_size,
        log_progress=log_progress,
    )
    _log(
        log_progress,
        f"descriptor_pipeline: category stats complete for {len(category_stats_result['built_categories'])} categories",
    )

    samples_root = data_root / "processed" / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    failures: List[DescriptorPipelineFailure] = []
    generated_count = 0

    _log(log_progress, f"descriptor_pipeline: materializing sample folders for {len(records)} records")
    for index, record in enumerate(records, start=1):
        sample_dir = samples_root / sample_folder_name(record)
        try:
            if skip_existing and _sample_folder_complete(sample_dir, record):
                if _should_log_sample_progress(index, len(records)):
                    _log(log_progress, f"descriptor_pipeline: skip existing sample {index}/{len(records)} {sample_dir.name}")
                continue
            if _should_log_sample_progress(index, len(records)):
                _log(log_progress, f"descriptor_pipeline: processing sample {index}/{len(records)} {sample_dir.name}")
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            materialize_sample_folder(record, data_root=data_root, sample_dir=sample_dir, target_size=target_size)
            generated_count += 1
        except Exception as exc:
            shutil.rmtree(sample_dir, ignore_errors=True)
            _log(
                log_progress,
                f"descriptor_pipeline: failed sample {sample_dir.name}: {type(exc).__name__}: {exc}",
            )
            failures.append(
                DescriptorPipelineFailure(
                    category=record.category,
                    sample_id=record.sample_id,
                    split=record.split,
                    defect_label=record.defect_label,
                    stage="sample_materialization",
                    message=repr(exc),
                )
            )
            if not continue_on_error:
                raise

    _log(log_progress, "descriptor_pipeline: validating generated descriptors")
    validation_summary = validate_descriptor_root(samples_root)
    validation_csv = data_root / "processed" / "reports" / "descriptor_validation.csv"
    validation_json = data_root / "processed" / "reports" / "descriptor_validation.json"
    write_validation_reports(validation_summary, validation_csv, validation_json)

    _log(log_progress, "descriptor_pipeline: auditing descriptor quality")
    quality_summary = audit_descriptor_root(samples_root)
    quality_csv = data_root / "processed" / "reports" / "descriptor_quality.csv"
    quality_json = data_root / "processed" / "reports" / "descriptor_quality_summary.json"
    write_descriptor_audit_reports(quality_summary, quality_csv, quality_json)

    _log(log_progress, "descriptor_pipeline: writing failure report")
    failure_csv = data_root / "processed" / "reports" / "descriptor_pipeline_failures.csv"
    failure_json = data_root / "processed" / "reports" / "descriptor_pipeline_failures.json"
    _write_failures(failures, failure_csv, failure_json)

    pipeline_summary = {
        "selected_records": len(records),
        "generated": {"category_stats": len(category_stats_result["built_categories"]), "samples": generated_count},
        "failure_count": len(failures),
        "samples_root": str(samples_root),
        "quality_json": str(quality_json),
        "validation_json": str(validation_json),
        "validation_summary": {
            "checked_manifests": validation_summary["checked_manifests"],
            "error_count": validation_summary["error_count"],
            "warning_count": validation_summary["warning_count"],
        },
    }
    pipeline_summary_path = data_root / "processed" / "reports" / "descriptor_pipeline_summary.json"
    pipeline_summary_path.write_text(json.dumps(pipeline_summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "records": records,
        "generated": pipeline_summary["generated"],
        "failures": failures,
        "pipeline_summary_path": pipeline_summary_path,
        "samples_root": samples_root,
        "quality_summary": quality_summary,
        "category_stats": category_stats_result,
    }


def sample_folder_name(record: SampleRecord) -> str:
    return f"{record.category}__{record.split}__{record.defect_label}__{_format_sample_id(record.sample_id)}"


def _log(log_progress: Callable[[str], None] | None, message: str) -> None:
    if log_progress is not None:
        log_progress(message)


def _should_log_sample_progress(index: int, total: int) -> bool:
    if total <= 10:
        return True
    if index in {1, total}:
        return True
    return index % 25 == 0


def materialize_sample_folder(
    record: SampleRecord,
    *,
    data_root: Path | str = "data",
    sample_dir: Path | str,
    target_size: tuple[int, int] = (256, 256),
) -> None:
    data_root = Path(data_root)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    stats_dir = data_root / "processed" / "category_stats" / record.category
    ir_stats = load_ir_category_stats(stats_dir)
    pointcloud_stats = load_pointcloud_category_stats(stats_dir)

    rgb_path = sample_dir / "rgb.png"
    _save_rgb_image(record.rgb_path, rgb_path, target_size=target_size)

    ir_outputs = generate_ir_sample_descriptors(record.ir_path, stats=ir_stats, target_size=target_size)
    save_ir_png(ir_outputs["ir_normalized"], sample_dir / "ir_normalized.png")
    save_ir_png(ir_outputs["ir_gradient"], sample_dir / "ir_gradient.png")
    save_ir_png(ir_outputs["ir_variance"], sample_dir / "ir_variance.png")
    save_ir_png(ir_outputs["ir_hotspot"], sample_dir / "ir_hotspot.png")

    pointcloud_outputs = generate_pointcloud_sample_descriptors(
        record.pointcloud_path,
        stats=pointcloud_stats,
        projection_size=target_size,
    )
    save_ir_png(pointcloud_outputs["pc_depth"], sample_dir / "pc_depth.png")
    save_ir_png(pointcloud_outputs["pc_density"], sample_dir / "pc_density.png")
    save_ir_png(pointcloud_outputs["pc_curvature"], sample_dir / "pc_curvature.png")
    save_ir_png(pointcloud_outputs["pc_roughness"], sample_dir / "pc_roughness.png")
    save_ir_png(pointcloud_outputs["pc_normal_deviation"], sample_dir / "pc_normal_deviation.png")

    crossmodal_outputs = generate_crossmodal_maps(
        ir_hotspot=ir_outputs["ir_hotspot"],
        ir_gradient=ir_outputs["ir_gradient"],
        pc_curvature=pointcloud_outputs["pc_curvature"],
        pc_roughness=pointcloud_outputs["pc_roughness"],
        ir_normalized=ir_outputs["ir_normalized"],
        pc_depth=pointcloud_outputs["pc_density"],
        target_size=target_size,
    )
    save_crossmodal_png(crossmodal_outputs["cross_ir_support"], sample_dir / "cross_ir_support.png")
    save_crossmodal_png(crossmodal_outputs["cross_geo_support"], sample_dir / "cross_geo_support.png")
    save_crossmodal_png(crossmodal_outputs["cross_agreement"], sample_dir / "cross_agreement.png")
    save_crossmodal_preview_png(crossmodal_outputs["cross_agreement"], sample_dir / "cross_agreement_preview.png")
    save_crossmodal_png(crossmodal_outputs["cross_inconsistency"], sample_dir / "cross_inconsistency.png")

    gt_mask_path = None
    if record.split == "test" and record.defect_label != "good" and record.rgb_gt_path:
        gt_mask_path = sample_dir / "gt_mask.png"
        _save_mask_image(record.rgb_gt_path, gt_mask_path, target_size=target_size)

    global_payload = {
        **ir_outputs["global"],
        **pointcloud_outputs["global"],
        **crossmodal_outputs["global"],
    }
    (sample_dir / "global.json").write_text(json.dumps(global_payload, indent=2, sort_keys=True), encoding="utf-8")

    meta_payload = {
        "category": record.category,
        "split": record.split,
        "defect_label": record.defect_label,
        "sample_id": _format_sample_id(record.sample_id),
        "sample_name": sample_dir.name,
        "rgb_source_path": record.rgb_path,
        "ir_source_path": record.ir_path,
        "pc_source_path": record.pointcloud_path,
        "gt_mask_path": str(gt_mask_path) if gt_mask_path is not None else None,
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_sample_id(sample_id: str) -> str:
    return sample_id.zfill(3) if sample_id.isdigit() else sample_id


def _sample_folder_complete(sample_dir: Path, record: SampleRecord) -> bool:
    required = {
        "rgb.png",
        "ir_normalized.png",
        "ir_gradient.png",
        "ir_variance.png",
        "ir_hotspot.png",
        "pc_depth.png",
        "pc_density.png",
        "pc_curvature.png",
        "pc_roughness.png",
        "pc_normal_deviation.png",
        "cross_ir_support.png",
        "cross_geo_support.png",
        "cross_agreement.png",
        "cross_inconsistency.png",
        "global.json",
        "meta.json",
    }
    if record.split == "test" and record.defect_label != "good":
        required.add("gt_mask.png")
    return sample_dir.exists() and all((sample_dir / filename).exists() for filename in required)


def _save_rgb_image(source_path: str, out_path: Path, *, target_size: tuple[int, int]) -> None:
    image = Image.open(source_path).convert("RGB").resize(target_size, Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _save_mask_image(source_path: str, out_path: Path, *, target_size: tuple[int, int]) -> None:
    image = Image.open(source_path).convert("L").resize(target_size, Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _write_failures(failures: Sequence[DescriptorPipelineFailure], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(DescriptorPipelineFailure.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in failures)
    json_path.write_text(json.dumps([asdict(item) for item in failures], indent=2, sort_keys=True), encoding="utf-8")
