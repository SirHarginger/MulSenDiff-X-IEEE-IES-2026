import json
from pathlib import Path

from src.preprocessing.dataset import build_dataset_index
from src.preprocessing.descriptor_pipeline import run_descriptor_pipeline


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
    b"\x3a\x7e\x9b\x55"
    b"\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


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


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MINIMAL_PNG)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_dataset(root: Path) -> Path:
    raw_root = root / "raw" / "MulSen_AD" / "widget"
    _write_png(raw_root / "RGB" / "train" / "0.png")
    _write_png(raw_root / "Infrared" / "train" / "0.png")
    _write_text(raw_root / "Pointcloud" / "train" / "0.stl", ASCII_STL)
    return root / "raw" / "MulSen_AD"


def test_run_descriptor_pipeline_writes_summary_and_index(tmp_path: Path) -> None:
    raw_root = _build_dataset(tmp_path)
    build_dataset_index(raw_root=raw_root, data_root=tmp_path)

    result = run_descriptor_pipeline(
        data_root=tmp_path,
        index_csv=tmp_path / "processed" / "manifests" / "dataset_index.csv",
        skip_existing=False,
        continue_on_error=False,
    )

    assert result["generated"]["infrared"] == 1
    assert result["generated"]["pointcloud"] == 1
    assert result["generated"]["crossmodal"] == 1
    assert result["pipeline_summary_path"].exists()
    summary = json.loads(result["pipeline_summary_path"].read_text(encoding="utf-8"))
    assert summary["failure_count"] == 0
