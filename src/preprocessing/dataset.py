from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.project_layout import resolve_processed_root


MODALITY_SPECS = {
    "RGB": {"sample_ext": ".png", "gt_ext": ".png"},
    "Infrared": {"sample_ext": ".png", "gt_ext": ".png"},
    "Pointcloud": {"sample_ext": ".stl", "gt_ext": ".txt"},
}

DEFAULT_PROCESSED_DIRS = (
    "manifests",
    "manifests/per_category",
    "category_stats",
    "samples",
    "reports",
)


@dataclass(frozen=True)
class SampleRecord:
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
    rgb_gt_csv_path: str
    rgb_gt_available: bool
    ir_gt_available: bool
    pointcloud_gt_available: bool
    rgb_declared_in_metadata: Optional[bool]
    infrared_declared_in_metadata: Optional[bool]
    pointcloud_declared_in_metadata: Optional[bool]


@dataclass(frozen=True)
class AlignmentIssue:
    category: str
    split: str
    defect_label: str
    sample_id: str
    issue: str


def _bool_from_csv(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    if value in {"1", "true", "True"}:
        return True
    if value in {"0", "false", "False"}:
        return False
    return None


def ensure_processed_layout(
    data_root: Path | str,
    *,
    processed_root: Path | str | None = None,
) -> Dict[str, Path]:
    data_root = Path(data_root)
    processed_root = resolve_processed_root(data_root=data_root, processed_root=processed_root)
    created: Dict[str, Path] = {}
    for relative_dir in DEFAULT_PROCESSED_DIRS:
        path = processed_root / relative_dir
        path.mkdir(parents=True, exist_ok=True)
        created[relative_dir] = path
    return created


def _list_categories(raw_root: Path) -> List[Path]:
    return sorted([path for path in raw_root.iterdir() if path.is_dir()])


def _read_rgb_gt_metadata(csv_path: Path) -> Dict[str, Dict[str, Optional[bool]]]:
    if not csv_path.exists():
        return {}

    metadata: Dict[str, Dict[str, Optional[bool]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            object_id = row.get("object")
            if object_id is None:
                continue
            metadata[str(object_id)] = {
                "RGB": _bool_from_csv(row.get("RGB")),
                "Infrared": _bool_from_csv(row.get("infrared")),
                "Pointcloud": _bool_from_csv(row.get("pointcloud")),
            }
    return metadata


def _build_train_record(
    category: str,
    sample_id: str,
    rgb_path: Path,
    ir_path: Path,
    pointcloud_path: Path,
) -> SampleRecord:
    return SampleRecord(
        category=category,
        sample_id=sample_id,
        split="train",
        defect_label="good",
        is_anomalous=False,
        rgb_path=str(rgb_path),
        ir_path=str(ir_path),
        pointcloud_path=str(pointcloud_path),
        rgb_gt_path="",
        ir_gt_path="",
        pointcloud_gt_path="",
        rgb_gt_csv_path="",
        rgb_gt_available=False,
        ir_gt_available=False,
        pointcloud_gt_available=False,
        rgb_declared_in_metadata=None,
        infrared_declared_in_metadata=None,
        pointcloud_declared_in_metadata=None,
    )


def _build_test_record(
    category: str,
    sample_id: str,
    defect_label: str,
    rgb_path: Path,
    ir_path: Path,
    pointcloud_path: Path,
    rgb_gt_path: Path,
    ir_gt_path: Path,
    pointcloud_gt_path: Path,
    rgb_gt_csv_path: Path,
    rgb_gt_metadata: Dict[str, Dict[str, Optional[bool]]],
) -> SampleRecord:
    per_object_metadata = rgb_gt_metadata.get(sample_id, {})
    return SampleRecord(
        category=category,
        sample_id=sample_id,
        split="test",
        defect_label=defect_label,
        is_anomalous=defect_label != "good",
        rgb_path=str(rgb_path),
        ir_path=str(ir_path),
        pointcloud_path=str(pointcloud_path),
        rgb_gt_path=str(rgb_gt_path) if rgb_gt_path.exists() else "",
        ir_gt_path=str(ir_gt_path) if ir_gt_path.exists() else "",
        pointcloud_gt_path=str(pointcloud_gt_path) if pointcloud_gt_path.exists() else "",
        rgb_gt_csv_path=str(rgb_gt_csv_path) if rgb_gt_csv_path.exists() else "",
        rgb_gt_available=rgb_gt_path.exists(),
        ir_gt_available=ir_gt_path.exists(),
        pointcloud_gt_available=pointcloud_gt_path.exists(),
        rgb_declared_in_metadata=per_object_metadata.get("RGB"),
        infrared_declared_in_metadata=per_object_metadata.get("Infrared"),
        pointcloud_declared_in_metadata=per_object_metadata.get("Pointcloud"),
    )


def _collect_train_records(category_path: Path) -> Tuple[List[SampleRecord], List[AlignmentIssue]]:
    category = category_path.name
    issues: List[AlignmentIssue] = []
    records: List[SampleRecord] = []

    train_files = {}
    for modality, spec in MODALITY_SPECS.items():
        train_dir = category_path / modality / "train"
        train_files[modality] = {path.stem: path for path in sorted(train_dir.glob(f"*{spec['sample_ext']}"))}

    sample_ids = sorted(set(train_files["RGB"]) | set(train_files["Infrared"]) | set(train_files["Pointcloud"]))
    for sample_id in sample_ids:
        missing = [modality for modality in MODALITY_SPECS if sample_id not in train_files[modality]]
        if missing:
            issues.append(
                AlignmentIssue(category, "train", "good", sample_id, f"missing_train_modalities={','.join(missing)}")
            )
            continue
        records.append(
            _build_train_record(
                category=category,
                sample_id=sample_id,
                rgb_path=train_files["RGB"][sample_id],
                ir_path=train_files["Infrared"][sample_id],
                pointcloud_path=train_files["Pointcloud"][sample_id],
            )
        )
    return records, issues


def _collect_test_records(category_path: Path) -> Tuple[List[SampleRecord], List[AlignmentIssue]]:
    category = category_path.name
    issues: List[AlignmentIssue] = []
    records: List[SampleRecord] = []

    rgb_test_root = category_path / "RGB" / "test"
    if not rgb_test_root.exists():
        return records, issues
    defect_labels = sorted([path.name for path in rgb_test_root.iterdir() if path.is_dir()])

    for defect_label in defect_labels:
        modality_files = {}
        for modality, spec in MODALITY_SPECS.items():
            label_dir = category_path / modality / "test" / defect_label
            modality_files[modality] = {path.stem: path for path in sorted(label_dir.glob(f"*{spec['sample_ext']}"))}

        rgb_gt_path = category_path / "RGB" / "GT" / defect_label
        rgb_gt_csv_path = rgb_gt_path / "data.csv"
        rgb_gt_metadata = _read_rgb_gt_metadata(rgb_gt_csv_path)

        sample_ids = sorted(set(modality_files["RGB"]) | set(modality_files["Infrared"]) | set(modality_files["Pointcloud"]))
        for sample_id in sample_ids:
            missing = [modality for modality in MODALITY_SPECS if sample_id not in modality_files[modality]]
            if missing:
                issues.append(
                    AlignmentIssue(
                        category,
                        "test",
                        defect_label,
                        sample_id,
                        f"missing_test_modalities={','.join(missing)}",
                    )
                )
                continue

            rgb_gt_file = category_path / "RGB" / "GT" / defect_label / f"{sample_id}{MODALITY_SPECS['RGB']['gt_ext']}"
            ir_gt_file = category_path / "Infrared" / "GT" / defect_label / f"{sample_id}{MODALITY_SPECS['Infrared']['gt_ext']}"
            pointcloud_gt_file = (
                category_path
                / "Pointcloud"
                / "GT"
                / defect_label
                / f"{sample_id}{MODALITY_SPECS['Pointcloud']['gt_ext']}"
            )

            records.append(
                _build_test_record(
                    category=category,
                    sample_id=sample_id,
                    defect_label=defect_label,
                    rgb_path=modality_files["RGB"][sample_id],
                    ir_path=modality_files["Infrared"][sample_id],
                    pointcloud_path=modality_files["Pointcloud"][sample_id],
                    rgb_gt_path=rgb_gt_file,
                    ir_gt_path=ir_gt_file,
                    pointcloud_gt_path=pointcloud_gt_file,
                    rgb_gt_csv_path=rgb_gt_csv_path,
                    rgb_gt_metadata=rgb_gt_metadata,
                )
            )
    return records, issues


def build_dataset_index(
    raw_root: Path | str,
    data_root: Path | str = "data",
    *,
    processed_root: Path | str | None = None,
) -> Dict[str, object]:
    raw_root = Path(raw_root)
    data_root = Path(data_root)
    resolved_processed_root = resolve_processed_root(data_root=data_root, processed_root=processed_root)
    ensure_processed_layout(data_root, processed_root=resolved_processed_root)

    records: List[SampleRecord] = []
    issues: List[AlignmentIssue] = []

    for category_path in _list_categories(raw_root):
        train_records, train_issues = _collect_train_records(category_path)
        test_records, test_issues = _collect_test_records(category_path)
        records.extend(train_records)
        records.extend(test_records)
        issues.extend(train_issues)
        issues.extend(test_issues)

    manifests_root = resolved_processed_root / "manifests"
    reports_root = resolved_processed_root / "reports"

    _write_index_files(records, manifests_root)
    _write_per_category_files(records, manifests_root / "per_category")
    _write_issue_report(issues, reports_root / "missing_alignment.csv")
    _write_audit_report(records, issues, reports_root / "data_audit.json")

    return {
        "records": records,
        "issues": issues,
        "manifest_csv": manifests_root / "dataset_index.csv",
        "manifest_jsonl": manifests_root / "dataset_index.jsonl",
        "audit_report": reports_root / "data_audit.json",
        "issue_report": reports_root / "missing_alignment.csv",
        "processed_root": resolved_processed_root,
    }


def _write_index_files(records: Sequence[SampleRecord], manifests_root: Path) -> None:
    manifests_root.mkdir(parents=True, exist_ok=True)
    csv_path = manifests_root / "dataset_index.csv"
    jsonl_path = manifests_root / "dataset_index.jsonl"

    rows = [asdict(record) for record in records]
    fieldnames = list(rows[0].keys()) if rows else list(SampleRecord.__dataclass_fields__.keys())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_per_category_files(records: Sequence[SampleRecord], per_category_root: Path) -> None:
    per_category_root.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)

    for category, category_records in grouped.items():
        out_path = per_category_root / f"{category}.csv"
        rows = [asdict(record) for record in category_records]
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _write_issue_report(issues: Sequence[AlignmentIssue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AlignmentIssue.__dataclass_fields__.keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(issue) for issue in issues)


def _write_audit_report(records: Sequence[SampleRecord], issues: Sequence[AlignmentIssue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_category = {}
    for record in records:
        stats = per_category.setdefault(
            record.category,
            {"train": 0, "test": 0, "anomalous_test": 0, "pointcloud_gt_available": 0, "rgb_gt_available": 0},
        )
        stats[record.split] += 1
        if record.is_anomalous:
            stats["anomalous_test"] += 1
        if record.pointcloud_gt_available:
            stats["pointcloud_gt_available"] += 1
        if record.rgb_gt_available:
            stats["rgb_gt_available"] += 1

    report = {
        "record_count": len(records),
        "issue_count": len(issues),
        "categories": sorted({record.category for record in records}),
        "per_category": per_category,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def iter_index(csv_path: Path | str) -> Iterable[SampleRecord]:
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield SampleRecord(
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
                rgb_gt_csv_path=row["rgb_gt_csv_path"],
                rgb_gt_available=row["rgb_gt_available"] == "True",
                ir_gt_available=row["ir_gt_available"] == "True",
                pointcloud_gt_available=row["pointcloud_gt_available"] == "True",
                rgb_declared_in_metadata=_bool_from_csv(row["rgb_declared_in_metadata"]),
                infrared_declared_in_metadata=_bool_from_csv(row["infrared_declared_in_metadata"]),
                pointcloud_declared_in_metadata=_bool_from_csv(row["pointcloud_declared_in_metadata"]),
            )
