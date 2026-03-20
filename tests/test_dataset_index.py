import csv
from pathlib import Path

from src.preprocessing.dataset import build_dataset_index, ensure_processed_layout


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_minimal_dataset(root: Path) -> Path:
    raw_root = root / "raw" / "MulSen_AD" / "widget"

    _touch(raw_root / "RGB" / "train" / "0.png")
    _touch(raw_root / "Infrared" / "train" / "0.png")
    _touch(raw_root / "Pointcloud" / "train" / "0.stl")

    _touch(raw_root / "RGB" / "test" / "good" / "0.png")
    _touch(raw_root / "Infrared" / "test" / "good" / "0.png")
    _touch(raw_root / "Pointcloud" / "test" / "good" / "0.stl")
    _touch(raw_root / "RGB" / "GT" / "good" / "data.csv", "object,RGB,infrared,pointcloud\n0,1,1,1\n")

    _touch(raw_root / "RGB" / "test" / "crack" / "1.png")
    _touch(raw_root / "Infrared" / "test" / "crack" / "1.png")
    _touch(raw_root / "Pointcloud" / "test" / "crack" / "1.stl")
    _touch(raw_root / "RGB" / "GT" / "crack" / "1.png")
    _touch(raw_root / "RGB" / "GT" / "crack" / "data.csv", "object,RGB,infrared,pointcloud\n1,1,1,0\n")
    _touch(raw_root / "Infrared" / "GT" / "crack" / "1.png")

    return root / "raw" / "MulSen_AD"


def test_ensure_processed_layout_creates_expected_directories(tmp_path: Path) -> None:
    created = ensure_processed_layout(tmp_path)
    assert (tmp_path / "processed" / "manifests").exists()
    assert (tmp_path / "processed" / "reports").exists()
    assert "manifests" in created


def test_build_dataset_index_writes_manifest_and_audit(tmp_path: Path) -> None:
    raw_root = _build_minimal_dataset(tmp_path)

    result = build_dataset_index(raw_root=raw_root, data_root=tmp_path)

    manifest_csv = result["manifest_csv"]
    audit_report = result["audit_report"]
    issue_report = result["issue_report"]

    assert manifest_csv.exists()
    assert audit_report.exists()
    assert issue_report.exists()

    rows = list(csv.DictReader(manifest_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 3

    crack_row = next(row for row in rows if row["sample_id"] == "1")
    assert crack_row["defect_label"] == "crack"
    assert crack_row["rgb_gt_available"] == "True"
    assert crack_row["ir_gt_available"] == "True"
    assert crack_row["pointcloud_gt_available"] == "False"
    assert crack_row["pointcloud_declared_in_metadata"] == "False"
