from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.preprocessing.dataset import SampleRecord, iter_index


DESCRIPTOR_TYPES = ("infrared", "pointcloud", "crossmodal")


@dataclass(frozen=True)
class DescriptorIndexRecord:
    category: str
    sample_id: str
    split: str
    defect_label: str
    is_anomalous: bool
    rgb_path: str
    ir_path: str
    pointcloud_path: str
    rgb_gt_path: str
    ir_gt_path: str
    pointcloud_gt_path: str
    infrared_manifest_path: str
    pointcloud_manifest_path: str
    crossmodal_manifest_path: str
    infrared_available: bool
    pointcloud_available: bool
    crossmodal_available: bool
    descriptor_ready: bool


def _manifest_path(data_root: Path, descriptor_type: str, record: SampleRecord) -> Path:
    return (
        data_root
        / "processed"
        / "descriptors"
        / descriptor_type
        / record.category
        / record.split
        / record.defect_label
        / record.sample_id
        / "manifest.json"
    )


def _record_key(record: SampleRecord) -> Tuple[str, str, str, str]:
    return (record.category, record.split, record.defect_label, record.sample_id)


def _descriptor_index_row(record: SampleRecord, data_root: Path) -> DescriptorIndexRecord:
    ir_manifest = _manifest_path(data_root, "infrared", record)
    pc_manifest = _manifest_path(data_root, "pointcloud", record)
    cm_manifest = _manifest_path(data_root, "crossmodal", record)
    return DescriptorIndexRecord(
        category=record.category,
        sample_id=record.sample_id,
        split=record.split,
        defect_label=record.defect_label,
        is_anomalous=record.is_anomalous,
        rgb_path=record.rgb_path,
        ir_path=record.ir_path,
        pointcloud_path=record.pointcloud_path,
        rgb_gt_path=record.rgb_gt_path,
        ir_gt_path=record.ir_gt_path,
        pointcloud_gt_path=record.pointcloud_gt_path,
        infrared_manifest_path=str(ir_manifest),
        pointcloud_manifest_path=str(pc_manifest),
        crossmodal_manifest_path=str(cm_manifest),
        infrared_available=ir_manifest.exists(),
        pointcloud_available=pc_manifest.exists(),
        crossmodal_available=cm_manifest.exists(),
        descriptor_ready=ir_manifest.exists() and pc_manifest.exists() and cm_manifest.exists(),
    )


def build_descriptor_index(
    *,
    data_root: Path | str = "data",
    sample_index_csv: Path | str = "data/processed/manifests/dataset_index.csv",
) -> Dict[str, object]:
    data_root = Path(data_root)
    sample_index_csv = Path(sample_index_csv)
    out_root = data_root / "processed" / "manifests"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [_descriptor_index_row(record, data_root) for record in iter_index(sample_index_csv)]
    csv_path = out_root / "descriptor_index.csv"
    jsonl_path = out_root / "descriptor_index.jsonl"
    coverage_json = data_root / "processed" / "reports" / "descriptor_coverage.json"
    coverage_csv = data_root / "processed" / "reports" / "descriptor_coverage.csv"

    _write_descriptor_index(rows, csv_path, jsonl_path)
    coverage = _write_descriptor_coverage(rows, coverage_json, coverage_csv)

    return {
        "rows": rows,
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "coverage_json": coverage_json,
        "coverage_csv": coverage_csv,
        "coverage": coverage,
    }


def _write_descriptor_index(rows: Sequence[DescriptorIndexRecord], csv_path: Path, jsonl_path: Path) -> None:
    payload = [asdict(row) for row in rows]
    fieldnames = list(payload[0].keys()) if payload else list(DescriptorIndexRecord.__dataclass_fields__.keys())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in payload:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_descriptor_coverage(
    rows: Sequence[DescriptorIndexRecord], coverage_json: Path, coverage_csv: Path
) -> Dict[str, object]:
    coverage_json.parent.mkdir(parents=True, exist_ok=True)
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)

    by_group: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in rows:
        key = (row.category, row.split)
        stats = by_group.setdefault(
            key,
            {
                "records": 0,
                "infrared_available": 0,
                "pointcloud_available": 0,
                "crossmodal_available": 0,
                "descriptor_ready": 0,
            },
        )
        stats["records"] += 1
        stats["infrared_available"] += int(row.infrared_available)
        stats["pointcloud_available"] += int(row.pointcloud_available)
        stats["crossmodal_available"] += int(row.crossmodal_available)
        stats["descriptor_ready"] += int(row.descriptor_ready)

    with coverage_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "category",
            "split",
            "records",
            "infrared_available",
            "pointcloud_available",
            "crossmodal_available",
            "descriptor_ready",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (category, split), stats in sorted(by_group.items()):
            writer.writerow({"category": category, "split": split, **stats})

    summary = {
        "record_count": len(rows),
        "ready_count": sum(int(row.descriptor_ready) for row in rows),
        "infrared_count": sum(int(row.infrared_available) for row in rows),
        "pointcloud_count": sum(int(row.pointcloud_available) for row in rows),
        "crossmodal_count": sum(int(row.crossmodal_available) for row in rows),
        "by_category_split": {
            f"{category}:{split}": stats for (category, split), stats in sorted(by_group.items())
        },
    }
    coverage_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def iter_descriptor_index(csv_path: Path | str) -> Iterable[DescriptorIndexRecord]:
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield DescriptorIndexRecord(
                category=row["category"],
                sample_id=row["sample_id"],
                split=row["split"],
                defect_label=row["defect_label"],
                is_anomalous=row["is_anomalous"] == "True",
                rgb_path=row["rgb_path"],
                ir_path=row["ir_path"],
                pointcloud_path=row["pointcloud_path"],
                rgb_gt_path=row["rgb_gt_path"],
                ir_gt_path=row["ir_gt_path"],
                pointcloud_gt_path=row["pointcloud_gt_path"],
                infrared_manifest_path=row["infrared_manifest_path"],
                pointcloud_manifest_path=row["pointcloud_manifest_path"],
                crossmodal_manifest_path=row["crossmodal_manifest_path"],
                infrared_available=row["infrared_available"] == "True",
                pointcloud_available=row["pointcloud_available"] == "True",
                crossmodal_available=row["crossmodal_available"] == "True",
                descriptor_ready=row["descriptor_ready"] == "True",
            )


class DescriptorBundleAccessor:
    def __init__(self, data_root: Path | str = ".") -> None:
        self.data_root = Path(data_root)

    def resolve(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        return path if path.is_absolute() else self.data_root / path

    def load_manifest(self, manifest_path: str | Path) -> List[Dict[str, object]]:
        return json.loads(self.resolve(manifest_path).read_text(encoding="utf-8"))

    def load_global_summary(self, manifest_path: str | Path) -> Dict[str, object]:
        manifest = self.load_manifest(manifest_path)
        global_artifact = next(artifact for artifact in manifest if artifact["kind"] == "global")
        return json.loads(self.resolve(global_artifact["path"]).read_text(encoding="utf-8"))

    def spatial_artifacts(self, manifest_path: str | Path) -> List[Path]:
        manifest = self.load_manifest(manifest_path)
        return [self.resolve(artifact["path"]) for artifact in manifest if artifact["kind"] == "spatial"]
