from pathlib import Path

from src.preprocessing.crossmodal_descriptors import (
    build_descriptor_metadata,
    descriptor_bundle_is_valid,
    validate_descriptor_bundle,
    write_descriptor_manifest,
)


def test_descriptor_bundle_validation_accepts_minimal_valid_bundle(tmp_path: Path) -> None:
    spatial_path = tmp_path / "spatial.npz"
    global_path = tmp_path / "global.npz"
    spatial_path.write_bytes(b"spatial")
    global_path.write_bytes(b"global")

    bundle = [
        build_descriptor_metadata(
            name="ir_spatial",
            kind="spatial",
            path=spatial_path,
            channels=3,
            width=640,
            height=480,
            dtype="float32",
        ),
        build_descriptor_metadata(
            name="global_summary",
            kind="global",
            path=global_path,
            vector_length=12,
            dtype="float32",
        ),
    ]

    issues = validate_descriptor_bundle(bundle)
    assert not [issue for issue in issues if issue.severity == "error"]
    assert descriptor_bundle_is_valid(bundle)


def test_descriptor_bundle_validation_flags_missing_spatial_shape(tmp_path: Path) -> None:
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"x")

    bundle = [
        build_descriptor_metadata(name="bad_spatial", kind="spatial", path=broken, channels=0, width=0, height=0)
    ]

    issues = validate_descriptor_bundle(bundle, require_global=False)
    messages = {issue.message for issue in issues}
    assert "invalid_spatial_shape" in messages
    assert "missing_spatial_channels" in messages


def test_descriptor_manifest_writer_creates_json(tmp_path: Path) -> None:
    payload = tmp_path / "descriptor.npy"
    payload.write_bytes(b"descriptor")
    manifest = tmp_path / "bundle.json"

    bundle = [
        build_descriptor_metadata(
            name="crossmodal",
            kind="global",
            path=payload,
            vector_length=4,
            dtype="float32",
        )
    ]
    write_descriptor_manifest(bundle, manifest)

    assert manifest.exists()
    assert '"name": "crossmodal"' in manifest.read_text(encoding="utf-8")
