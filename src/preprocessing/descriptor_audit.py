from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageStat

from src.preprocessing.crossmodal_descriptors import (
    DescriptorArtifact,
    DescriptorValidationIssue,
    validate_descriptor_bundle,
)


@dataclass(frozen=True)
class DescriptorQualityIssue:
    manifest_path: str
    artifact: str
    severity: str
    message: str
    value: str = ""


def _load_manifest(path: Path | str) -> List[DescriptorArtifact]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DescriptorArtifact(**item) for item in payload]


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _spatial_stats(path: Path) -> Dict[str, float]:
    image = Image.open(path).convert("L")
    stat = ImageStat.Stat(image)
    pixels = list(image.getdata())
    total = max(len(pixels), 1)
    nonzero_ratio = sum(1 for value in pixels if value > 0) / total
    minimum, maximum = image.getextrema()
    return {
        "mean": float(stat.mean[0]),
        "std": float(stat.stddev[0]),
        "nonzero_ratio": float(nonzero_ratio),
        "minimum": float(minimum),
        "maximum": float(maximum),
    }


def _artifact_name_group(name: str) -> str:
    if name.startswith("infrared_"):
        return "infrared"
    if name.startswith("pointcloud_"):
        return "pointcloud"
    if name.startswith("crossmodal_"):
        return "crossmodal"
    return "unknown"


def _audit_spatial_descriptor(
    manifest_path: Path, artifact: DescriptorArtifact, issues: List[DescriptorQualityIssue], stats_out: Dict[str, float]
) -> None:
    path = Path(artifact.path)
    stats = _spatial_stats(path)
    stats_out.update(stats)

    if stats["maximum"] == stats["minimum"]:
        issues.append(
            DescriptorQualityIssue(
                str(manifest_path), artifact.name, "warning", "constant_spatial_descriptor", _format_float(stats["mean"])
            )
        )

    if stats["std"] < 1.0:
        issues.append(
            DescriptorQualityIssue(
                str(manifest_path), artifact.name, "warning", "very_low_spatial_variance", _format_float(stats["std"])
            )
        )

    if stats["nonzero_ratio"] == 0.0:
        issues.append(
            DescriptorQualityIssue(str(manifest_path), artifact.name, "warning", "empty_spatial_signal", "0.0")
        )

    sparse_like = ("hotspot", "agreement", "edges", "support")
    if any(token in artifact.name for token in sparse_like):
        if stats["nonzero_ratio"] < 0.001:
            issues.append(
                DescriptorQualityIssue(
                    str(manifest_path),
                    artifact.name,
                    "warning",
                    "sparse_map_nearly_empty",
                    _format_float(stats["nonzero_ratio"]),
                )
            )
        if stats["nonzero_ratio"] > 0.95:
            issues.append(
                DescriptorQualityIssue(
                    str(manifest_path),
                    artifact.name,
                    "warning",
                    "sparse_map_nearly_full",
                    _format_float(stats["nonzero_ratio"]),
                )
            )


def _audit_global_descriptor(
    manifest_path: Path, artifact: DescriptorArtifact, issues: List[DescriptorQualityIssue], stats_out: Dict[str, float]
) -> None:
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            stats_out[key] = float(value)

    name = artifact.name
    if name == "infrared_global":
        hotspot_ratio = float(payload.get("hotspot_ratio", 0.0))
        std_intensity = float(payload.get("std_intensity", 0.0))
        if hotspot_ratio <= 0.0:
            issues.append(
                DescriptorQualityIssue(str(manifest_path), name, "warning", "hotspot_ratio_zero", _format_float(hotspot_ratio))
            )
        if std_intensity < 3.0:
            issues.append(
                DescriptorQualityIssue(
                    str(manifest_path), name, "warning", "infrared_low_global_contrast", _format_float(std_intensity)
                )
            )

    if name == "pointcloud_global":
        coverage_ratio = float(payload.get("coverage_ratio", 0.0))
        vertex_count = float(payload.get("vertex_count", 0.0))
        if vertex_count <= 0:
            issues.append(
                DescriptorQualityIssue(str(manifest_path), name, "error", "pointcloud_vertex_count_zero", str(vertex_count))
            )
        if coverage_ratio < 0.01:
            issues.append(
                DescriptorQualityIssue(
                    str(manifest_path), name, "warning", "pointcloud_projection_low_coverage", _format_float(coverage_ratio)
                )
            )

    if name == "crossmodal_global":
        agreement_std = float(payload.get("agreement_std", 0.0))
        support_mean = float(payload.get("support_mean", 0.0))
        if agreement_std < 1.0:
            issues.append(
                DescriptorQualityIssue(
                    str(manifest_path), name, "warning", "crossmodal_low_agreement_variation", _format_float(agreement_std)
                )
            )
        if support_mean <= 0.5:
            issues.append(
                DescriptorQualityIssue(
                    str(manifest_path), name, "warning", "crossmodal_low_support_signal", _format_float(support_mean)
                )
            )


def audit_descriptor_manifest(
    manifest_path: Path | str,
) -> Dict[str, object]:
    manifest_path = Path(manifest_path)
    artifacts = _load_manifest(manifest_path)

    issues: List[DescriptorQualityIssue] = []
    quality_stats: Dict[str, Dict[str, float]] = {}

    validation_issues = validate_descriptor_bundle(artifacts)
    for issue in validation_issues:
        issues.append(
            DescriptorQualityIssue(
                str(manifest_path),
                issue.artifact,
                issue.severity,
                f"validation::{issue.message}",
                "",
            )
        )

    for artifact in artifacts:
        per_artifact_stats: Dict[str, float] = {}
        if artifact.kind == "spatial":
            _audit_spatial_descriptor(manifest_path, artifact, issues, per_artifact_stats)
        elif artifact.kind == "global":
            _audit_global_descriptor(manifest_path, artifact, issues, per_artifact_stats)
        quality_stats[artifact.name] = per_artifact_stats

    return {
        "manifest_path": str(manifest_path),
        "artifacts": artifacts,
        "issues": issues,
        "quality_stats": quality_stats,
    }


def audit_descriptor_root(root: Path | str) -> Dict[str, object]:
    root = Path(root)
    manifest_paths = sorted(root.rglob("manifest.json"))
    issue_rows: List[DescriptorQualityIssue] = []
    numeric_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    manifests_with_errors = 0

    for manifest_path in manifest_paths:
        result = audit_descriptor_manifest(manifest_path)
        issues: Sequence[DescriptorQualityIssue] = result["issues"]  # type: ignore[assignment]
        if any(issue.severity == "error" for issue in issues):
            manifests_with_errors += 1
        issue_rows.extend(issues)

        quality_stats: Dict[str, Dict[str, float]] = result["quality_stats"]  # type: ignore[assignment]
        for artifact_name, stats in quality_stats.items():
            group = _artifact_name_group(artifact_name)
            for key, value in stats.items():
                numeric_stats[group][key].append(value)

    summary = {
        "checked_manifests": len(manifest_paths),
        "manifests_with_errors": manifests_with_errors,
        "error_count": sum(1 for issue in issue_rows if issue.severity == "error"),
        "warning_count": sum(1 for issue in issue_rows if issue.severity == "warning"),
        "per_group_means": {
            group: {key: round(mean(values), 6) for key, values in metrics.items() if values}
            for group, metrics in numeric_stats.items()
        },
        "issue_rows": issue_rows,
    }
    return summary


def write_descriptor_audit_reports(summary: Dict[str, object], csv_path: Path | str, json_path: Path | str) -> None:
    import csv

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
