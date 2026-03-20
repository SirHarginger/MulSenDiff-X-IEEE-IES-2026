from __future__ import annotations

from pathlib import Path

from src.explainer.evidence_builder import EvidencePackage


def render_templated_explanation(package: EvidencePackage) -> str:
    top_region = package.top_regions[0] if package.top_regions else {}
    centroid = top_region.get("centroid_xy", [0.0, 0.0])
    bbox = top_region.get("bbox_xyxy", [0, 0, 0, 0])

    lines = [
        f"Status: {package.status} ({package.severity_0_100:.1f}/100 severity, {package.confidence_0_100:.1f}/100 confidence)",
        f"Sample: {package.category} / {package.defect_label} / {package.sample_id}",
        f"Raw anomaly score: {package.raw_score:.6f}",
        f"Dominant region: centroid=({centroid[0]:.1f}, {centroid[1]:.1f}), bbox={bbox}, affected_area={package.affected_area_pct:.2f}%",
        "Observations:",
    ]
    lines.extend(f"- {item}" for item in package.rgb_observations)
    lines.extend(f"- {item}" for item in package.thermal_observations)
    lines.extend(f"- {item}" for item in package.geometric_observations)
    lines.extend(f"- {item}" for item in package.cross_modal_support)
    lines.append("Confidence Notes:")
    lines.extend(f"- {item}" for item in package.confidence_notes)
    lines.append("Evidence Breakdown:")
    for key, value in package.evidence_breakdown.items():
        lines.append(f"- {key}: {value:.3f}%")
    lines.append("Recommended Action:")
    lines.append(f"- Inspect the dominant region first and verify whether {package.score_label.lower()} aligns with visible or sensor-supported irregularities.")
    return "\n".join(lines)


def save_templated_explanation(text: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
