import json
from pathlib import Path

from PIL import Image

from src.preprocessing.descriptor_audit import audit_descriptor_root
from src.preprocessing.dataset import build_dataset_index
from src.preprocessing.descriptor_pipeline import run_descriptor_pipeline, sample_folder_name


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


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), value).save(path)


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (120, 80, 40)).save(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_dataset(root: Path) -> Path:
    raw_root = root / "raw" / "MulSen_AD" / "widget"
    _write_rgb(raw_root / "RGB" / "train" / "0.png")
    _write_png(raw_root / "Infrared" / "train" / "0.png", 120)
    _write_text(raw_root / "Pointcloud" / "train" / "0.stl", ASCII_STL)
    return root / "raw" / "MulSen_AD"


def test_run_descriptor_pipeline_writes_sample_folder_and_summary(tmp_path: Path) -> None:
    raw_root = _build_dataset(tmp_path)
    dataset_result = build_dataset_index(raw_root=raw_root, data_root=tmp_path)

    result = run_descriptor_pipeline(
        data_root=tmp_path,
        index_csv=dataset_result["manifest_csv"],
        skip_existing=False,
        continue_on_error=False,
    )

    assert result["generated"]["category_stats"] == 1
    assert result["generated"]["samples"] == 1
    assert result["pipeline_summary_path"].exists()
    summary = json.loads(result["pipeline_summary_path"].read_text(encoding="utf-8"))
    assert summary["failure_count"] == 0

    sample_dir = tmp_path / "processed" / "samples" / "widget__train__good__000"
    assert sample_dir.exists()
    assert (sample_dir / "rgb.png").exists()
    assert (sample_dir / "global.json").exists()
    assert (sample_dir / "meta.json").exists()
    assert (sample_dir / "cross_agreement_preview.png").exists()


def test_audit_descriptor_root_reports_incomplete_sample_dir_without_crashing(tmp_path: Path) -> None:
    sample_dir = tmp_path / "samples" / "widget__train__good__000"
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(sample_dir / "rgb.png")

    summary = audit_descriptor_root(tmp_path / "samples")

    assert summary["error_count"] == 1
    issue = summary["issue_rows"][0]
    assert issue.artifact == "meta.json"
    assert issue.message == "missing_metadata_file"


def test_run_descriptor_pipeline_cleans_partial_sample_dir_after_failure(tmp_path: Path, monkeypatch) -> None:
    raw_root = _build_dataset(tmp_path)
    dataset_result = build_dataset_index(raw_root=raw_root, data_root=tmp_path)

    def _boom(**kwargs):
        raise ValueError("forced crossmodal failure")

    monkeypatch.setattr("src.preprocessing.descriptor_pipeline.generate_crossmodal_maps", _boom)

    result = run_descriptor_pipeline(
        data_root=tmp_path,
        index_csv=dataset_result["manifest_csv"],
        skip_existing=False,
        continue_on_error=True,
    )

    sample_dir = tmp_path / "processed" / "samples" / "widget__train__good__000"
    assert len(result["failures"]) == 1
    assert not sample_dir.exists()


def test_run_descriptor_pipeline_keeps_sample_when_alignment_check_fails(tmp_path: Path, monkeypatch) -> None:
    raw_root = _build_dataset(tmp_path)
    dataset_result = build_dataset_index(raw_root=raw_root, data_root=tmp_path)

    def _failed_alignment(*args, **kwargs):
        return {
            "passed": False,
            "message": "boundary_overlap_exceeds_tolerance",
            "ir_bbox": (0, 0, 7, 7),
            "pc_bbox": (8, 8, 15, 15),
            "max_delta": 8,
        }

    monkeypatch.setattr("src.preprocessing.crossmodal_descriptors.verify_crossmodal_alignment", _failed_alignment)

    result = run_descriptor_pipeline(
        data_root=tmp_path,
        index_csv=dataset_result["manifest_csv"],
        skip_existing=False,
        continue_on_error=False,
    )

    sample_dir = tmp_path / "processed" / "samples" / "widget__train__good__000"
    payload = json.loads((sample_dir / "global.json").read_text(encoding="utf-8"))

    assert len(result["failures"]) == 0
    assert sample_dir.exists()
    assert payload["cross_alignment_passed"] is False
    assert payload["cross_alignment_message"] == "boundary_overlap_exceeds_tolerance"
