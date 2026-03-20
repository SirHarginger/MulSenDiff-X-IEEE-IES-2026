from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.explainer.evidence_builder import EvidencePackage


@dataclass(frozen=True)
class RetrievedContextItem:
    title: str
    snippet: str
    source: str


def retrieve_context_for_evidence(package: EvidencePackage) -> List[RetrievedContextItem]:
    """Placeholder hook for a future manufacturing-knowledge retriever.

    Guidance:
    - retrieval should be driven by the structured evidence package
    - queries should use category, dominant region cues, and modality evidence
    - retrieved text must remain inspectable before it is passed to an LLM
    """

    return []
