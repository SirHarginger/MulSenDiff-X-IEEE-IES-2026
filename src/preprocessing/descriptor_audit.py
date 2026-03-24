from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from src.preprocessing.crossmodal_descriptors import DescriptorArtifact, validate_descriptor_bundle


@dataclass(frozen=True)
class DescriptorQualityIssue:
    sample_dir: str
    artifact: str
    severity: str
    message: str
    value: str = ""


def audit_descriptor_manifest(manifest_path: Path | str) -> Dict[str, object]:
    manifest_path = Path(manifest_path)
    artifacts = [DescriptorArtifact(**item) for item in json.loads(manifest_path.read_text(encoding="utf-8"))]
    issues: List[DescriptorQualityIssue] = []
    quality_stats: Dict[str, Dict[str, float]] = {}

    for issue in validate_descriptor_bundle(artifacts):
        issues.append(
            DescriptorQualityIssue(
                sample_dir=str(manifest_path.parent),
                artifact=issue.artifact,
                severity=issue.severity,
                message=issue.message,
            )
        )

    for artifact in artifacts:
        path = Path(artifact.path)
        stats: Dict[str, float] = {}
        if artifact.kind == "spatial" and path.exists():
            image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            stats["mean"] = float(image.mean())
            stats["std"] = float(image.std())
            stats["nonzero_ratio"] = float(np.mean(image > 0))
            if stats["std"] < 1e-6:
                issues.append(
                    DescriptorQualityIssue(str(manifest_path.parent), artifact.name, "warning", "constant_spatial_descriptor", f"{stats['mean']:.6f}")
                )
        if artifact.kind == "global" and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    stats[key] = float(value)
            if "hotspot_ratio" in payload and float(payload["hotspot_ratio"]) <= 0.0:
                issues.append(
                    DescriptorQualityIssue(str(manifest_path.parent), artifact.name, "warning", "hotspot_ratio_zero", f"{float(payload['hotspot_ratio']):.6f}")
                )
            if "std_intensity" in payload and float(payload["std_intensity"]) < 3.0:
                issues.append(
                    DescriptorQualityIssue(str(manifest_path.parent), artifact.name, "warning", "infrared_low_global_contrast", f"{float(payload['std_intensity']):.6f}")
                )
        quality_stats[artifact.name] = stats

    return {
        "manifest_path": str(manifest_path),
        "artifacts": artifacts,
        "issues": issues,
        "quality_stats": quality_stats,
    }


def audit_descriptor_root(root: Path | str) -> Dict[str, object]:
    root = Path(root)
    manifest_paths = sorted(root.rglob("manifest.json")) if root.exists() else []
    if manifest_paths:
        issue_rows: List[DescriptorQualityIssue] = []
        numeric_stats: Dict[str, Dict[str, List[float]]] = {"pointcloud": {}, "infrared": {}, "crossmodal": {}}
        manifests_with_errors = 0
        for manifest_path in manifest_paths:
            result = audit_descriptor_manifest(manifest_path)
            issues: Sequence[DescriptorQualityIssue] = result["issues"]  # type: ignore[assignment]
            if any(issue.severity == "error" for issue in issues):
                manifests_with_errors += 1
            issue_rows.extend(issues)
            quality_stats: Dict[str, Dict[str, float]] = result["quality_stats"]  # type: ignore[assignment]
            for artifact_name, stats in quality_stats.items():
                group = "infrared" if artifact_name.startswith("infrared") else "pointcloud" if artifact_name.startswith("pointcloud") else "crossmodal"
                numeric_stats.setdefault(group, {})
                for key, value in stats.items():
                    numeric_stats[group].setdefault(key, []).append(value)
        return {
            "checked_manifests": len(manifest_paths),
            "manifests_with_errors": manifests_with_errors,
            "error_count": sum(1 for row in issue_rows if row.severity == "error"),
            "warning_count": sum(1 for row in issue_rows if row.severity == "warning"),
            "per_group_means": {
                group: {metric: round(mean(values), 6) for metric, values in metrics.items() if values}
                for group, metrics in numeric_stats.items()
                if metrics
            },
            "issue_rows": issue_rows,
        }

    sample_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    issue_rows: List[DescriptorQualityIssue] = []
    per_group_values: Dict[str, Dict[str, List[float]]] = {
        "infrared": {"hotspot_area_fraction": [], "ir_max_gradient": []},
        "pointcloud": {"pc_mean_curvature": [], "pc_max_roughness": [], "pc_p95_normal_deviation": []},
        "crossmodal": {"cross_overlap_score": [], "cross_inconsistency_score": []},
    }

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "meta.json"
        global_path = sample_dir / "global.json"

        if not meta_path.exists():
            issue_rows.append(
                DescriptorQualityIssue(str(sample_dir), "meta.json", "error", "missing_metadata_file")
            )
            continue
        if not global_path.exists():
            issue_rows.append(
                DescriptorQualityIssue(str(sample_dir), "global.json", "error", "missing_global_summary")
            )
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issue_rows.append(
                DescriptorQualityIssue(str(sample_dir), "meta.json", "error", "invalid_metadata_json", str(exc))
            )
            continue

        try:
            global_payload = json.loads(global_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issue_rows.append(
                DescriptorQualityIssue(str(sample_dir), "global.json", "error", "invalid_global_summary_json", str(exc))
            )
            continue

        try:
            ir_hotspot_area_fraction = float(global_payload["ir_hotspot_area_fraction"])
            ir_max_gradient = float(global_payload["ir_max_gradient"])
            pc_mean_curvature = float(global_payload["pc_mean_curvature"])
            pc_max_roughness = float(global_payload["pc_max_roughness"])
            pc_p95_normal_deviation = float(global_payload["pc_p95_normal_deviation"])
            cross_overlap_score = float(global_payload["cross_overlap_score"])
            cross_inconsistency_score = float(global_payload["cross_inconsistency_score"])
        except (KeyError, TypeError, ValueError) as exc:
            issue_rows.append(
                DescriptorQualityIssue(str(sample_dir), "global.json", "error", "invalid_global_summary_payload", str(exc))
            )
            continue

        per_group_values["infrared"]["hotspot_area_fraction"].append(ir_hotspot_area_fraction)
        per_group_values["infrared"]["ir_max_gradient"].append(ir_max_gradient)
        per_group_values["pointcloud"]["pc_mean_curvature"].append(pc_mean_curvature)
        per_group_values["pointcloud"]["pc_max_roughness"].append(pc_max_roughness)
        per_group_values["pointcloud"]["pc_p95_normal_deviation"].append(pc_p95_normal_deviation)
        per_group_values["crossmodal"]["cross_overlap_score"].append(cross_overlap_score)
        per_group_values["crossmodal"]["cross_inconsistency_score"].append(cross_inconsistency_score)

        if meta.get("split") == "train" and meta.get("defect_label") == "good" and ir_hotspot_area_fraction >= 0.02:
            issue_rows.append(
                DescriptorQualityIssue(
                    str(sample_dir),
                    "global.json",
                    "warning",
                    "train_hotspot_area_fraction_high",
                    f"{ir_hotspot_area_fraction:.6f}",
                )
            )

        for artifact in ["cross_agreement.png", "cross_inconsistency.png", "pc_density.png"]:
            path = sample_dir / artifact
            if not path.exists():
                continue
            image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            if image.max() == image.min():
                issue_rows.append(
                    DescriptorQualityIssue(
                        str(sample_dir),
                        artifact,
                        "warning",
                        "constant_map",
                        f"{float(image.mean()):.6f}",
                    )
                )
            if artifact == "pc_density.png" and float(image.max()) <= 0.0:
                issue_rows.append(
                    DescriptorQualityIssue(str(sample_dir), artifact, "error", "empty_density_projection", "0.000000")
                )

    return {
        "checked_manifests": len(sample_dirs),
        "manifests_with_errors": sum(1 for row in issue_rows if row.severity == "error"),
        "error_count": sum(1 for row in issue_rows if row.severity == "error"),
        "warning_count": sum(1 for row in issue_rows if row.severity == "warning"),
        "per_group_means": {
            group: {metric: round(mean(values), 6) for metric, values in metrics.items() if values}
            for group, metrics in per_group_values.items()
        },
        "issue_rows": issue_rows,
    }


def write_descriptor_audit_reports(summary: Dict[str, object], csv_path: Path | str, json_path: Path | str) -> None:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows: Sequence[DescriptorQualityIssue] = summary["issue_rows"]  # type: ignore[assignment]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DescriptorQualityIssue.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    payload = dict(summary)
    payload["issue_rows"] = [asdict(row) for row in rows]
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
