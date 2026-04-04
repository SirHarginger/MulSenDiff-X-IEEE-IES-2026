from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.explainer.evidence_builder import EvidencePackage
from src.explainer.llm_pipeline import (
    GeminiExplanationProvider,
    GeminiProviderConfig,
    build_detector_block,
    generate_operator_report,
    load_gemini_provider_config,
    save_explanation_bundle,
)
from src.explainer.retriever import RetrievedContextItem

GeminiConfig = GeminiProviderConfig


def load_gemini_config(env: Mapping[str, str] | None = None) -> GeminiConfig:
    return load_gemini_provider_config(env)


def build_root_cause_evidence(
    *,
    package: EvidencePackage,
    regions: Sequence[Mapping[str, Any]],
    evidence_bullets: Sequence[str],
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
    reference_comparison: str | None = None,
) -> Dict[str, Any]:
    return build_detector_block(
        package=package,
        regions=regions,
        evidence_bullets=evidence_bullets,
        retrieved_context=retrieved_context,
        reference_comparison=reference_comparison,
    )


def generate_root_cause_explanation(
    package: EvidencePackage,
    *,
    config: GeminiConfig,
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
    evidence_bullets: Sequence[str] | None = None,
    reference_comparison: str | None = None,
) -> Dict[str, Any]:
    provider = GeminiExplanationProvider(config) if config.enabled else None
    return generate_operator_report(
        package=package,
        retrieved_context=retrieved_context,
        regions=regions,
        evidence_bullets=evidence_bullets,
        reference_comparison=reference_comparison,
        provider=provider,
    )


def save_gemini_generation(
    *,
    output_root: Path | str,
    sample_slug: str,
    source_eval_run: Path | str,
    source_evidence_path: Path | str,
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]],
    prompt_payload: Dict[str, Any],
    generation: Dict[str, Any],
) -> Dict[str, str]:
    del sample_slug, retrieved_context, prompt_payload
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = output_root / "operator_report_bundle.json"
    save_explanation_bundle(
        output_path=bundle_path,
        package=package,
        generation=generation,
        source_eval_run=source_eval_run,
        source_evidence_path=source_evidence_path,
    )
    markdown_path = output_root / "operator_report.md"
    markdown_path.write_text(str(generation.get("markdown", "")), encoding="utf-8")
    return {
        "session_root": str(output_root),
        "markdown_path": str(markdown_path),
        "json_path": str(bundle_path),
    }
