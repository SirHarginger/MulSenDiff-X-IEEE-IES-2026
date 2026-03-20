from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from src.preprocessing.crossmodal_descriptors import DescriptorArtifact, DescriptorValidationIssue, validate_descriptor_bundle


def _load_manifest(path: Path | str) -> List[DescriptorArtifact]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DescriptorArtifact(**item) for item in payload]


def validate_descriptor_root(root: Path | str) -> Dict[str, object]:
    root = Path(root)
    manifest_paths = sorted(root.rglob("manifest.json"))
    rows = []
    error_count = 0
    warning_count = 0

    for manifest_path in manifest_paths:
        artifacts = _load_manifest(manifest_path)
        issues: Sequence[DescriptorValidationIssue] = validate_descriptor_bundle(artifacts)
        for issue in issues:
            if issue.severity == "error":
                error_count += 1
            elif issue.severity == "warning":
                warning_count += 1
            rows.append(
                {
                    "manifest_path": str(manifest_path),
                    "artifact": issue.artifact,
                    "severity": issue.severity,
                    "message": issue.message,
                }
            )

    return {
        "checked_manifests": len(manifest_paths),
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
        fieldnames = ["manifest_path", "artifact", "severity", "message"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = dict(summary)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
