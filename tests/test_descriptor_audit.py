import json
from pathlib import Path

from src.preprocessing.crossmodal_descriptors import build_descriptor_metadata, write_descriptor_manifest
from src.preprocessing.descriptor_audit import audit_descriptor_manifest, audit_descriptor_root


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
    b"\x3a\x7e\x9b\x55"
    b"\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MINIMAL_PNG)


def test_audit_descriptor_manifest_flags_low_signal_global(tmp_path: Path) -> None:
    spatial_path = tmp_path / "hotspot.png"
    global_path = tmp_path / "global.json"
    manifest_path = tmp_path / "manifest.json"

    _write_png(spatial_path)
    global_path.write_text(
        json.dumps(
            {
                "source_path": "x",
                "width": 1,
                "height": 1,
                "mean_intensity": 0.0,
                "std_intensity": 0.0,
                "hotspot_threshold": 255,
                "hotspot_ratio": 0.0,
            }
        ),
        encoding="utf-8",
    )

    write_descriptor_manifest(
        [
            build_descriptor_metadata(
                name="infrared_hotspot",
                kind="spatial",
                path=spatial_path,
                channels=1,
                width=1,
                height=1,
                dtype="uint8",
            ),
            build_descriptor_metadata(
                name="infrared_global",
                kind="global",
                path=global_path,
                vector_length=4,
                dtype="json",
            ),
        ],
        manifest_path,
    )

    result = audit_descriptor_manifest(manifest_path)
    messages = {issue.message for issue in result["issues"]}
    assert "hotspot_ratio_zero" in messages
    assert "infrared_low_global_contrast" in messages


def test_audit_descriptor_root_summarizes_manifests(tmp_path: Path) -> None:
    spatial_path = tmp_path / "bundle" / "edges.png"
    global_path = tmp_path / "bundle" / "global.json"
    manifest_path = tmp_path / "bundle" / "manifest.json"

    _write_png(spatial_path)
    global_path.write_text(
        json.dumps({"vertex_count": 1, "coverage_ratio": 0.25, "source_path": "x"}),
        encoding="utf-8",
    )
    write_descriptor_manifest(
        [
            build_descriptor_metadata(
                name="pointcloud_depth_edges",
                kind="spatial",
                path=spatial_path,
                channels=1,
                width=1,
                height=1,
                dtype="uint8",
            ),
            build_descriptor_metadata(
                name="pointcloud_global",
                kind="global",
                path=global_path,
                vector_length=5,
                dtype="json",
            ),
        ],
        manifest_path,
    )

    summary = audit_descriptor_root(tmp_path)
    assert summary["checked_manifests"] == 1
    assert "pointcloud" in summary["per_group_means"]
