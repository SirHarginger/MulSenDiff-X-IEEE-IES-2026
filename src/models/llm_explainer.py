from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.explainer.evidence_builder import EvidencePackage


@dataclass(frozen=True)
class ExplanationOutput:
    headline: str
    observations: List[str]
    hypothesis: str
    recommended_action: str
    metadata: Dict[str, str]


def build_llm_explanation(package: EvidencePackage) -> ExplanationOutput:
    """Future hook for grounded LLM generation.

    Guidance:
    - never prompt from raw anomaly maps or raw tensors directly
    - use the saved evidence package plus optional retrieved context
    - keep observations separate from hypotheses
    - expose all generated text alongside the evidence payload for review
    """

    raise NotImplementedError(
        "LLM explanation is not implemented yet. Use the templated evidence report first."
    )
