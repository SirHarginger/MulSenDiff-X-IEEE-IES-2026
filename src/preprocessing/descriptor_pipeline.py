from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from src.preprocessing.crossmodal_descriptors import generate_crossmodal_descriptor_bundle
from src.preprocessing.dataset import SampleRecord, iter_index
from src.preprocessing.descriptor_audit import audit_descriptor_root, write_descriptor_audit_reports
from src.preprocessing.descriptor_dataset import build_descriptor_index
from src.preprocessing.ir_descriptors import generate_ir_descriptor_bundle
from src.preprocessing.pointcloud_projector import generate_pointcloud_descriptor_bundle
from src.preprocessing.descriptor_validation import validate_descriptor_root, write_validation_reports


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


def _bundle_dir(data_root: Path, descriptor_type: str, record: SampleRecord) -> Path:
    return (
        data_root
        / "processed"
        / "descriptors"
        / descriptor_type
        / record.category
        / record.split
        / record.defect_label
        / record.sample_id
    )


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

    failures: List[DescriptorPipelineFailure] = []
    generated = {"infrared": 0, "pointcloud": 0, "crossmodal": 0}

    for record in records:
        try:
            ir_dir = _bundle_dir(data_root, "infrared", record)
            ir_manifest = ir_dir / "manifest.json"
            if not (skip_existing and ir_manifest.exists()):
                generate_ir_descriptor_bundle(record.ir_path, ir_dir)
                generated["infrared"] += 1

            pc_dir = _bundle_dir(data_root, "pointcloud", record)
            pc_manifest = pc_dir / "manifest.json"
            if not (skip_existing and pc_manifest.exists()):
                generate_pointcloud_descriptor_bundle(record.pointcloud_path, pc_dir)
                generated["pointcloud"] += 1

            cm_dir = _bundle_dir(data_root, "crossmodal", record)
            cm_manifest = cm_dir / "manifest.json"
            if not (skip_existing and cm_manifest.exists()):
                if ir_manifest.exists() and pc_manifest.exists():
                    generate_crossmodal_descriptor_bundle(record.rgb_path, ir_manifest, pc_manifest, cm_dir)
                    generated["crossmodal"] += 1
                else:
                    failures.append(
                        DescriptorPipelineFailure(
                            record.category,
                            record.sample_id,
                            record.split,
                            record.defect_label,
                            "crossmodal",
                            "missing_required_input_manifests",
                        )
                    )
        except Exception as exc:
            failures.append(
                DescriptorPipelineFailure(
                    record.category,
                    record.sample_id,
                    record.split,
                    record.defect_label,
                    "generation",
                    repr(exc),
                )
            )
            if not continue_on_error:
                raise

    descriptor_index = build_descriptor_index(data_root=data_root, sample_index_csv=index_csv)

    validation_summary = validate_descriptor_root(data_root / "processed" / "descriptors")
    validation_csv = data_root / "processed" / "reports" / "descriptor_validation.csv"
    validation_json = data_root / "processed" / "reports" / "descriptor_validation.json"
    write_validation_reports(validation_summary, validation_csv, validation_json)

    quality_summary = audit_descriptor_root(data_root / "processed" / "descriptors")
    quality_csv = data_root / "processed" / "reports" / "descriptor_quality.csv"
    quality_json = data_root / "processed" / "reports" / "descriptor_quality_summary.json"
    write_descriptor_audit_reports(quality_summary, quality_csv, quality_json)

    failure_csv = data_root / "processed" / "reports" / "descriptor_pipeline_failures.csv"
    failure_json = data_root / "processed" / "reports" / "descriptor_pipeline_failures.json"
    _write_failures(failures, failure_csv, failure_json)

    pipeline_summary = {
        "selected_records": len(records),
        "generated": generated,
        "failure_count": len(failures),
        "descriptor_index_csv": str(descriptor_index["csv_path"]),
        "coverage_json": str(descriptor_index["coverage_json"]),
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
        "generated": generated,
        "failures": failures,
        "descriptor_index": descriptor_index,
        "quality_summary": quality_summary,
        "pipeline_summary_path": pipeline_summary_path,
    }


def _write_failures(failures: Sequence[DescriptorPipelineFailure], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(DescriptorPipelineFailure.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in failures)
    json_path.write_text(json.dumps([asdict(item) for item in failures], indent=2, sort_keys=True), encoding="utf-8")
