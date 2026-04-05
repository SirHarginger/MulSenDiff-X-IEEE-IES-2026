from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.explainer.evidence_builder import EvidencePackage
from src.explainer.llm_pipeline import (
    OperatorReport,
    build_explainer_context_pack,
    build_detector_only_report,
    determine_explanation_state,
    render_operator_report_markdown,
    save_explanation_bundle as save_rag_explanation_bundle,
)
from src.explainer.retriever import RetrievedContextItem

ExplanationOutput = OperatorReport


def build_prompt_payload(
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem],
) -> Dict[str, Any]:
    return build_explainer_context_pack(
        package=package,
        retrieved_context=retrieved_context,
    )


def build_llm_explanation(
    package: EvidencePackage,
    *,
    retrieved_context: Sequence[RetrievedContextItem] | None = None,
    prompt_payload: Dict[str, Any] | None = None,
) -> ExplanationOutput:
    recommended_state = determine_explanation_state(
        package,
        (prompt_payload or {}).get("block_b_retrieved_support", retrieved_context or []),
    )
    return build_detector_only_report(package, recommended_state=recommended_state)


def render_llm_explanation_markdown(output: ExplanationOutput) -> str:
    return render_operator_report_markdown(output)


def save_explanation_bundle(
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem],
    prompt_payload: Dict[str, Any],
    explanation: ExplanationOutput,
    path: Path | str,
) -> Path:
    generation = {
        "provider": "detector_fallback",
        "model": "",
        "provider_available": False,
        "provider_error": "",
        "used_fallback": True,
        "text": "",
        "structured_output": explanation.to_dict(),
        "explanation_state": explanation.explanation_state,
        "markdown": render_operator_report_markdown(explanation),
        "context_pack": prompt_payload or build_prompt_payload(package, list(retrieved_context)),
    }
    return save_rag_explanation_bundle(
        output_path=path,
        package=package,
        generation=generation,
    )
