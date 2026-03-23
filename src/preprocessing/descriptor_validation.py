from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


TRAIN_REQUIRED_FILES = [
    "rgb.png",
    "ir_normalized.png",
    "ir_gradient.png",
    "ir_variance.png",
    "ir_hotspot.png",
    "pc_depth.png",
    "pc_density.png",
    "pc_curvature.png",
    "pc_roughness.png",
    "pc_normal_deviation.png",
    "cross_ir_support.png",
    "cross_geo_support.png",
    "cross_agreement.png",
    "cross_inconsistency.png",
    "global.json",
    "meta.json",
]


GLOBAL_FIELDS = {
    "ir_hotspot_area_fraction",
    "ir_max_gradient",
    "ir_mean_hotspot_intensity",
    "ir_hotspot_compactness",
    "pc_mean_curvature",
    "pc_max_roughness",
    "pc_p95_normal_deviation",
    "cross_overlap_score",
    "cross_centroid_distance",
    "cross_inconsistency_score",
}


def validate_descriptor_root(root: Path | str) -> Dict[str, object]:
    root = Path(root)
    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    rows: List[Dict[str, str]] = []
    error_count = 0
    warning_count = 0

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        required = list(TRAIN_REQUIRED_FILES)
        needs_gt = bool(meta.get("split") == "test" and meta.get("defect_label") not in {"", "good"})
        if needs_gt:
            required.append("gt_mask.png")

        for filename in required:
            path = sample_dir / filename
            if not path.exists():
                error_count += 1
                rows.append(
                    {
                        "sample_dir": str(sample_dir),
                        "artifact": filename,
                        "severity": "error",
                        "message": "missing_required_file",
                    }
                )
                continue
            if path.is_file() and path.stat().st_size <= 0:
                error_count += 1
                rows.append(
                    {
                        "sample_dir": str(sample_dir),
                        "artifact": filename,
                        "severity": "error",
                        "message": "empty_file",
                    }
                )

        if meta_path.exists():
            global_path = sample_dir / "global.json"
            if global_path.exists():
                payload = json.loads(global_path.read_text(encoding="utf-8"))
                missing_fields = sorted(GLOBAL_FIELDS - set(payload.keys()))
                for field in missing_fields:
                    error_count += 1
                    rows.append(
                        {
                            "sample_dir": str(sample_dir),
                            "artifact": "global.json",
                            "severity": "error",
                            "message": f"missing_global_field::{field}",
                        }
                    )
            if needs_gt and not meta.get("gt_mask_path"):
                warning_count += 1
                rows.append(
                    {
                        "sample_dir": str(sample_dir),
                        "artifact": "meta.json",
                        "severity": "warning",
                        "message": "missing_gt_mask_path_metadata",
                    }
                )

    return {
        "checked_manifests": len(sample_dirs),
        "error_count": error_count,
        "warning_count": warning_count,
        "rows": rows,
    }


def write_validation_reports(summary: Dict[str, object], csv_path: Path | str, json_path: Path | str) -> None:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = summary["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_dir", "artifact", "severity", "message"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
