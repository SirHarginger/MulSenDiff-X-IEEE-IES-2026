import csv
import json
from pathlib import Path

from src.preprocessing.dataset import build_dataset_index
from src.preprocessing.descriptor_dataset import (
    DescriptorBundleAccessor,
    build_descriptor_index,
    iter_descriptor_index,
)


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
    b"\x3a\x7e\x9b\x55"
    b"\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_minimal_dataset(root: Path) -> Path:
    raw_root = root / "raw" / "MulSen_AD" / "widget"
    (raw_root / "RGB" / "train").mkdir(parents=True, exist_ok=True)
    (raw_root / "Infrared" / "train").mkdir(parents=True, exist_ok=True)
    (raw_root / "Pointcloud" / "train").mkdir(parents=True, exist_ok=True)
    _touch(raw_root / "RGB" / "train" / "0.png")
    _touch(raw_root / "Infrared" / "train" / "0.png")
    _touch(raw_root / "Pointcloud" / "train" / "0.stl")
    return root / "raw" / "MulSen_AD"


def test_build_descriptor_index_tracks_manifest_availability(tmp_path: Path) -> None:
    raw_root = _build_minimal_dataset(tmp_path)
    build_dataset_index(raw_root=raw_root, data_root=tmp_path)

    ir_manifest = (
        tmp_path / "processed" / "descriptors" / "infrared" / "widget" / "train" / "good" / "0" / "manifest.json"
    )
    ir_manifest.parent.mkdir(parents=True, exist_ok=True)
    ir_manifest.write_text(json.dumps([]), encoding="utf-8")

    result = build_descriptor_index(
        data_root=tmp_path,
        sample_index_csv=tmp_path / "processed" / "manifests" / "dataset_index.csv",
    )
    rows = list(iter_descriptor_index(result["csv_path"]))
    assert len(rows) == 1
    assert rows[0].infrared_available is True
    assert rows[0].pointcloud_available is False
    assert rows[0].descriptor_ready is False


def test_descriptor_bundle_accessor_loads_global_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    global_json = tmp_path / "global.json"
    global_json.write_text(json.dumps({"support_mean": 1.23}), encoding="utf-8")
    manifest.write_text(
        json.dumps([{"name": "crossmodal_global", "kind": "global", "path": str(global_json)}]),
        encoding="utf-8",
    )

    accessor = DescriptorBundleAccessor(tmp_path)
    summary = accessor.load_global_summary(manifest)
    assert summary["support_mean"] == 1.23
