from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.explainer.evidence_builder import EvidencePackage
from src.explainer.retriever import RetrievedContextItem


@dataclass(frozen=True)
class ExplanationOutput:
    detection_statement: str
    localisation: str
    evidence_explanation: str
    root_cause_and_action: str
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_prompt_payload(
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem],
) -> Dict[str, Any]:
    return {
        "task": "industrial_anomaly_explanation",
        "style": "concise_auditable_report",
        "required_sections": [
            "detection_statement",
            "localisation",
            "evidence_explanation",
            "root_cause_and_action",
        ],
        "evidence": package.to_dict(),
        "retrieved_context": [item.to_dict() for item in retrieved_context],
    }


def build_llm_explanation(
    package: EvidencePackage,
    *,
    retrieved_context: Sequence[RetrievedContextItem] | None = None,
    prompt_payload: Dict[str, Any] | None = None,
) -> ExplanationOutput:
    context_items = list(retrieved_context or [])
    top_region = package.top_regions[0] if package.top_regions else {}
    centroid = top_region.get("centroid_xy", [0.0, 0.0])
    bbox = top_region.get("bbox_xyxy", [0, 0, 0, 0])

    detection_statement = (
        f"{package.status}: sample score {package.raw_score:.4f} with "
        f"{package.confidence_0_100:.1f}/100 confidence."
    )
    localisation = (
        f"The dominant region is centered near ({centroid[0]:.1f}, {centroid[1]:.1f}) "
        f"with bounding box {bbox} and covers {package.affected_area_pct:.2f}% of the object area."
    )
    evidence_sentences = [
        *package.rgb_observations,
        *package.thermal_observations,
        *package.geometric_observations,
        *package.cross_modal_support,
    ]
    evidence_explanation = " ".join(sentence.rstrip(".") + "." for sentence in evidence_sentences)
    if context_items:
        evidence_explanation += (
            f" Retrieved context suggests: {context_items[0].title} - {context_items[0].snippet}"
        )

    if package.global_descriptor_score >= 0.5:
        root_cause = "Sample-level descriptor drift supports a broader physical deviation rather than a purely visual artifact."
    else:
        root_cause = "The anomaly appears localized and should be verified against the dominant thermal/geometric cues first."
    root_cause_and_action = (
        f"{root_cause} Recommended action: inspect the highlighted region, compare it with the supporting sensor cues, "
        "and confirm whether the suspected mechanism matches the observed defect pattern."
    )

    metadata = {
        "engine": "internal_rule_debug_backend",
        "context_items": str(len(context_items)),
        "prompt_sections": str(len((prompt_payload or {}).get("required_sections", []))),
    }
    return ExplanationOutput(
        detection_statement=detection_statement,
        localisation=localisation,
        evidence_explanation=evidence_explanation,
        root_cause_and_action=root_cause_and_action,
        metadata=metadata,
    )


def render_llm_explanation_markdown(output: ExplanationOutput) -> str:
    lines = [
        "# Detection Statement",
        output.detection_statement,
        "",
        "# Localisation",
        output.localisation,
        "",
        "# Evidence Explanation",
        output.evidence_explanation,
        "",
        "# Root Cause And Action",
        output.root_cause_and_action,
    ]
    return "\n".join(lines)


def save_explanation_bundle(
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem],
    prompt_payload: Dict[str, Any],
    explanation: ExplanationOutput,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence_package": package.to_dict(),
        "retrieved_context": [item.to_dict() for item in retrieved_context],
        "prompt_payload": prompt_payload,
        "explanation_output": explanation.to_dict(),
        "rendered_markdown": render_llm_explanation_markdown(explanation),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
