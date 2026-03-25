import csv
import json
from pathlib import Path

from src.synthetic.generator import RECIPES, generate_synthetic_anomaly_bundles
from tests.test_data_loader_runtime import _build_descriptor_ready_dataset


def test_generate_synthetic_anomaly_bundles_runtime(tmp_path: Path) -> None:
    root = _build_descriptor_ready_dataset(tmp_path)
    result = generate_synthetic_anomaly_bundles(
        samples_root=root / "processed" / "samples",
        output_root=root / "synthetic",
        recipes=RECIPES,
        limit=1,
    )

    manifest_csv = Path(result["manifest_csv"])
    index_json = Path(result["index_json"])
    assert manifest_csv.exists()
    assert index_json.exists()

    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(RECIPES)
    for row in rows:
        assert Path(row["rgb_path"]).exists()
        assert Path(row["evidence_path"]).exists()
        assert Path(row["provenance_path"]).exists()

    index = json.loads(index_json.read_text(encoding="utf-8"))
    assert len(index) == len(RECIPES)
    assert not (root / "data" / "splits" / "synthetic_manifest.csv").exists()
