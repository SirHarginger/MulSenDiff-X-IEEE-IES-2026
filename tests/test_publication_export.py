import csv
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from src.publication.export import export_publication_package


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)


def _build_eval_run(
    root: Path,
    *,
    per_category: dict[str, dict[str, float | int | str]],
    evaluation: dict[str, float | int | str],
    image_score_rows: list[dict[str, object]],
) -> Path:
    _write_json(root / "summary.json", {"checkpoint_path": str(root / "checkpoints" / "best.pt"), "run_dir": str(root)})
    _write_json(root / "metrics" / "evaluation.json", evaluation)
    _write_json(root / "metrics" / "per_category.json", per_category)
    _write_json(root / "metrics" / "image_score_data.json", image_score_rows)
    _write_image(root / "plots" / "image_roc_curve.png", (255, 0, 0))
    _write_image(root / "plots" / "image_pr_curve.png", (0, 255, 0))
    _write_image(root / "plots" / "image_score_distribution.png", (0, 0, 255))
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints" / "best.pt").write_bytes(b"fake")
    return root


def test_export_publication_package_core_outputs(tmp_path: Path) -> None:
    sample_root = tmp_path / "processed" / "samples" / "capsule__test__crack__000"
    rgb_path = sample_root / "rgb.png"
    overlay_path = sample_root / "overlay.png"
    ir_map_path = sample_root / "ir_hotspot.png"
    aux_map_path = sample_root / "cross_agreement.png"
    for path, color in [
        (rgb_path, (180, 80, 80)),
        (overlay_path, (200, 120, 40)),
        (ir_map_path, (140, 20, 20)),
        (aux_map_path, (20, 80, 160)),
    ]:
        _write_image(path, color)

    package_path = tmp_path / "evidence" / "capsule_case.json"
    _write_json(
        package_path,
        {
            "source_paths": {
                "rgb_path": str(rgb_path),
                "overlay_path": str(overlay_path),
            }
        },
    )

    shared_per_category = {
        "capsule": {
            "image_auroc": 0.92,
            "pixel_auroc": 0.81,
            "pixel_aupr": 0.77,
            "pixel_f1_max": 0.7,
            "aupro": 0.74,
            "good_false_positive_rate": 0.1,
            "anomalous_normal_rate": 0.05,
        },
        "screw": {
            "image_auroc": 0.71,
            "pixel_auroc": 0.65,
            "pixel_aupr": 0.58,
            "pixel_f1_max": 0.51,
            "aupro": 0.55,
            "good_false_positive_rate": 0.2,
            "anomalous_normal_rate": 0.12,
        },
    }
    shared_eval = _build_eval_run(
        tmp_path / "shared_eval",
        per_category=shared_per_category,
        evaluation={
            "image_auroc": 0.84,
            "macro_image_auroc": 0.815,
            "std_image_auroc": 0.105,
            "worst_image_auroc": 0.71,
            "worst_image_auroc_category": "screw",
            "worst5_image_auroc": 0.815,
            "pixel_auroc": 0.73,
            "macro_pixel_auroc": 0.73,
            "pixel_aupr": 0.675,
            "macro_pixel_aupr": 0.675,
            "pixel_f1_max": 0.605,
            "macro_pixel_f1_max": 0.605,
            "aupro": 0.645,
            "macro_aupro": 0.645,
        },
        image_score_rows=[
            {"category": "capsule", "label": 0, "score": 0.10},
            {"category": "capsule", "label": 1, "score": 0.90},
            {"category": "screw", "label": 0, "score": 0.20},
            {"category": "screw", "label": 1, "score": 0.80},
        ],
    )
    shared_nocat_eval = _build_eval_run(
        tmp_path / "shared_nocat_eval",
        per_category={
            "capsule": dict(shared_per_category["capsule"], image_auroc=0.87),
            "screw": dict(shared_per_category["screw"], image_auroc=0.62),
        },
        evaluation={
            "image_auroc": 0.77,
            "macro_image_auroc": 0.745,
            "std_image_auroc": 0.125,
            "worst_image_auroc": 0.62,
            "worst_image_auroc_category": "screw",
            "worst5_image_auroc": 0.745,
            "pixel_auroc": 0.69,
            "macro_pixel_auroc": 0.69,
            "pixel_aupr": 0.62,
            "macro_pixel_aupr": 0.62,
            "pixel_f1_max": 0.57,
            "macro_pixel_f1_max": 0.57,
            "aupro": 0.6,
            "macro_aupro": 0.6,
        },
        image_score_rows=[
            {"category": "capsule", "label": 0, "score": 0.15},
            {"category": "capsule", "label": 1, "score": 0.86},
            {"category": "screw", "label": 0, "score": 0.32},
            {"category": "screw", "label": 1, "score": 0.72},
        ],
    )
    percat_capsule_eval = _build_eval_run(
        tmp_path / "percat_capsule_eval",
        per_category={"capsule": shared_per_category["capsule"]},
        evaluation={"image_auroc": 0.93, "pixel_auroc": 0.81, "pixel_aupr": 0.77, "pixel_f1_max": 0.7, "aupro": 0.74},
        image_score_rows=[
            {"category": "capsule", "label": 0, "score": 0.08},
            {"category": "capsule", "label": 1, "score": 0.91},
        ],
    )
    percat_screw_eval = _build_eval_run(
        tmp_path / "percat_screw_eval",
        per_category={"screw": shared_per_category["screw"]},
        evaluation={"image_auroc": 0.69, "pixel_auroc": 0.65, "pixel_aupr": 0.58, "pixel_f1_max": 0.51, "aupro": 0.55},
        image_score_rows=[
            {"category": "screw", "label": 0, "score": 0.28},
            {"category": "screw", "label": 1, "score": 0.71},
        ],
    )

    percat_manifest = tmp_path / "percat_eval_manifest.csv"
    _write_csv(
        percat_manifest,
        [
            {"category": "capsule", "eval_run_dir": str(percat_capsule_eval)},
            {"category": "screw", "eval_run_dir": str(percat_screw_eval)},
        ],
    )
    qualitative_manifest = tmp_path / "qualitative_examples.csv"
    _write_csv(
        qualitative_manifest,
        [
            {
                "figure_id": "figure1",
                "package_path": str(package_path),
                "thermal_map_name": "ir_hotspot",
                "auxiliary_map_name": "cross_agreement",
                "title": "Example anomaly",
                "caption": "A compact multimodal defect case.",
            }
        ],
    )

    outputs = export_publication_package(
        shared_eval_run=shared_eval,
        shared_nocat_eval_run=shared_nocat_eval,
        percat_eval_manifest=percat_manifest,
        qualitative_manifest=qualitative_manifest,
        out_dir=tmp_path / "publication",
    )

    assert Path(outputs["table1_main_comparison"]).exists()
    assert Path(outputs["table2_per_category_shared"]).exists()
    assert Path(outputs["figure1_multimodal_case"]).exists()
    assert Path(outputs["figure2_category_stability"]).exists()
    workbook_path = Path(outputs["publication_tables_xlsx"])
    assert workbook_path.exists()

    with ZipFile(workbook_path, "r") as archive:
        names = set(archive.namelist())
    assert "xl/worksheets/sheet1.xml" in names
    assert "xl/worksheets/sheet2.xml" in names

    table1_rows = list(csv.DictReader(Path(outputs["table1_main_comparison"]).open("r", encoding="utf-8", newline="")))
    assert [row["protocol"] for row in table1_rows] == ["Shared", "PerCat", "Shared-NoCat"]

    caption_path = Path(outputs["figure1_multimodal_case"]).with_suffix(".caption.txt")
    assert caption_path.exists()
    assert "compact multimodal defect case" in caption_path.read_text(encoding="utf-8").lower()
