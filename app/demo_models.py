from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from PIL import Image

from src.explainer.evidence_builder import EvidencePackage


@dataclass(frozen=True)
class InspectionVisuals:
    original_image: Image.Image
    overlay_image: Image.Image
    reference_image: Image.Image | None = None
    panel_path: str = ""


@dataclass(frozen=True)
class EvidenceSummary:
    package: EvidencePackage
    regions: List[Dict[str, Any]]
    bullets: List[str]


@dataclass(frozen=True)
class RetrievalStatus:
    configured: bool
    chunk_count: int
    label: str
    context_count: int = 0
    items: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return self.context_count > 0


@dataclass(frozen=True)
class ExplanationBundle:
    mode_label: str
    explanation_state: str
    report: Dict[str, Any]
    error_message: str
    provider_available: bool
    provider_name: str
    model_name: str
    used_fallback: bool
    retrieval: RetrievalStatus
    context_pack: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectionBundle:
    selected_record_name: str
    visuals: InspectionVisuals
    evidence: EvidenceSummary
    explanation: ExplanationBundle
    debug_payload: Dict[str, Any] = field(default_factory=dict)
    report_markdown: str = ""
    session_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


def retrieval_items_to_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            payload.append(dict(item))
        elif hasattr(item, "to_dict"):
            payload.append(dict(item.to_dict()))
        else:
            payload.append({"title": str(item)})
    return payload
