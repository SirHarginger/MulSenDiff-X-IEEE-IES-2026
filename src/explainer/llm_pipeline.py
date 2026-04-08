from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Sequence

from src.explainer.evidence_builder import EvidencePackage, build_retrieval_query
from src.explainer.retriever import RetrievedContextItem


@dataclass(frozen=True)
class GeminiProviderConfig:
    api_key: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)


@dataclass(frozen=True)
class SupportingCitation:
    title: str
    source: str
    snippet: str
    relevance_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorReport:
    explanation_state: str
    likely_cause: str
    why_flagged: str
    recommended_action: str
    operator_summary: str
    supporting_citations: list[SupportingCitation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["supporting_citations"] = [citation.to_dict() for citation in self.supporting_citations]
        return payload


class ExplanationProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(
        self,
        detector_block: Mapping[str, Any],
        retrieved_block: Sequence[Mapping[str, Any]],
        schema_block: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ...


class GeminiExplanationProvider:
    provider_name = "gemini"

    def __init__(self, config: GeminiProviderConfig) -> None:
        self.config = config
        self.model_name = config.model

    def generate(
        self,
        detector_block: Mapping[str, Any],
        retrieved_block: Sequence[Mapping[str, Any]],
        schema_block: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError(
                "Gemini is not configured. Add your key to config/gemini.local.json "
                "or set GOOGLE_API_KEY / GEMINI_API_KEY and rerun the inspection."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "The `google-genai` package is not installed. Run `pip install google-genai` in your app environment."
            ) from exc

        prompt_sections = [
            "You are an industrial anomaly explanation assistant.",
            "Use only the three blocks below.",
            "Detector evidence is the source of truth for anomaly status.",
            "Retrieved support may refine likely cause and action only when compatible with detector evidence.",
            "Do not invent citations. Do not claim external support unless it is in Block B.",
            "",
            "Block A: Detector Evidence",
            json.dumps(detector_block, indent=2, sort_keys=True),
            "",
            "Block B: Retrieved Support",
            json.dumps(list(retrieved_block), indent=2, sort_keys=True),
            "",
            "Block C: Output Schema And Grounding Rules",
            json.dumps(dict(schema_block), indent=2, sort_keys=True),
        ]
        prompt = "\n".join(prompt_sections)
        client = genai.Client(api_key=self.config.api_key)
        response = client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema_block["response_schema"],
                "temperature": 0.2,
            },
        )
        text = str(getattr(response, "text", "") or "").strip()
        return {
            "text": text,
            "structured_output": _parse_structured_json_response(text),
            "provider": self.provider_name,
            "model": self.model_name,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_local_gemini_json_config() -> Dict[str, str]:
    candidates = [
        _repo_root() / "config" / "gemini.local.json",
        _repo_root() / "config" / "gemini.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse {path}. Check the JSON syntax."
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Expected {path} to contain a JSON object.")
        return {
            "api_key": str(
                payload.get("api_key")
                or payload.get("google_api_key")
                or payload.get("gemini_api_key")
                or ""
            ).strip(),
            "model": str(
                payload.get("model")
                or payload.get("gemini_model")
                or ""
            ).strip(),
        }
    return {}


def _load_local_gemini_provider_config() -> Dict[str, str]:
    try:
        module = import_module("src.explainer.local_gemini_config")
    except ModuleNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(
            "Failed to import src/explainer/local_gemini_config.py. "
            "Check the file for syntax errors."
        ) from exc

    api_key = str(
        getattr(module, "GOOGLE_API_KEY", "")
        or getattr(module, "GEMINI_API_KEY", "")
        or getattr(module, "MULSENDIFFX_GEMINI_API_KEY", "")
        or ""
    ).strip()
    model = str(
        getattr(module, "GEMINI_MODEL", "")
        or getattr(module, "MULSENDIFFX_GEMINI_MODEL", "")
        or ""
    ).strip()
    return {
        "api_key": api_key,
        "model": model,
    }


def load_gemini_provider_config(env: Mapping[str, str] | None = None) -> GeminiProviderConfig:
    resolved_env = dict(env or os.environ)
    local_json_config = _load_local_gemini_json_config()
    local_config = _load_local_gemini_provider_config()
    api_key = str(
        local_json_config.get("api_key")
        or local_config.get("api_key")
        or resolved_env.get("GOOGLE_API_KEY")
        or resolved_env.get("GEMINI_API_KEY")
        or resolved_env.get("MULSENDIFFX_GEMINI_API_KEY")
        or ""
    ).strip()
    model = str(
        local_json_config.get("model")
        or local_config.get("model")
        or resolved_env.get("MULSENDIFFX_GEMINI_MODEL")
        or resolved_env.get("GEMINI_MODEL")
        or "gemini-2.5-flash"
    ).strip()
    return GeminiProviderConfig(api_key=api_key, model=model)


def build_detector_block(
    *,
    package: EvidencePackage,
    regions: Sequence[Mapping[str, Any]] | None = None,
    evidence_bullets: Sequence[str] | None = None,
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
    reference_comparison: str | None = None,
) -> Dict[str, Any]:
    localized_regions = _serialize_regions(regions or [], package)
    context_payload = [_serialize_retrieved_context_item(item) for item in (retrieved_context or [])]
    return {
        "status": "normal" if package.status.lower().startswith("normal") else "anomalous",
        "category": package.category,
        "defect_label": package.defect_label,
        "sample_id": package.sample_id,
        "score_mode": package.score_mode,
        "score_label": package.score_label,
        "anomaly_score": round(float(package.raw_score), 6),
        "severity": {
            "label": package.status,
            "value_0_100": round(float(package.severity_0_100), 3),
            "band": str(package.retrieval_features.severity_band),
        },
        "confidence": {
            "value_0_100": round(float(package.confidence_0_100), 3),
            "band": str(package.retrieval_features.confidence_band),
            "recommended_explanation_state": determine_explanation_state(package, context_payload),
        },
        "affected_area_percent": round(float(package.affected_area_pct), 3),
        "localized_regions": localized_regions,
        "evidence_breakdown": dict(package.evidence_breakdown),
        "observations": {
            "rgb": list(package.rgb_observations),
            "thermal": list(package.thermal_observations),
            "geometric": list(package.geometric_observations),
            "cross_modal": list(package.cross_modal_support),
        },
        "evidence_bullets": list(evidence_bullets or _default_evidence_bullets(package)),
        "confidence_notes": list(package.confidence_notes),
        "uncertainty_signals": _uncertainty_signals(package),
        "normal_reference_comparison": reference_comparison or _reference_comparison_from_context(context_payload),
        "retrieval_features": build_retrieval_query(package),
        "provenance": {
            "object_score_strategy": str(package.provenance.get("object_score_strategy", "")),
            "internal_defect_gate_fired": bool(package.provenance.get("internal_defect_gate_fired", False)),
            "internal_defect_gate_mode": str(package.provenance.get("internal_defect_gate_mode", "")),
            "score_mode": str(package.provenance.get("score_mode", package.score_mode)),
        },
    }


def build_schema_block() -> Dict[str, Any]:
    return {
        "grounding_rules": [
            "Detector evidence is the source of truth for anomaly status.",
            "Retrieved support may refine likely cause or action only if compatible with detector evidence.",
            "Do not invent causes, manuals, or citations.",
            "If Block B is empty, supporting_citations must be an empty list.",
            "If support is weak or absent, explanation_state must be cautious or abstained.",
        ],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "explanation_state": {
                    "type": "STRING",
                    "enum": ["grounded", "cautious", "abstained"],
                },
                "likely_cause": {"type": "STRING"},
                "why_flagged": {"type": "STRING"},
                "recommended_action": {"type": "STRING"},
                "operator_summary": {"type": "STRING"},
                "supporting_citations": {
                    "type": "ARRAY",
                    "maxItems": 3,
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "title": {"type": "STRING"},
                            "source": {"type": "STRING"},
                            "snippet": {"type": "STRING"},
                            "relevance_reason": {"type": "STRING"},
                        },
                        "required": ["title", "source", "snippet", "relevance_reason"],
                    },
                },
            },
            "required": [
                "explanation_state",
                "likely_cause",
                "why_flagged",
                "recommended_action",
                "operator_summary",
                "supporting_citations",
            ],
        },
    }


def build_explainer_context_pack(
    *,
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
    evidence_bullets: Sequence[str] | None = None,
    reference_comparison: str | None = None,
) -> Dict[str, Any]:
    detector_block = build_detector_block(
        package=package,
        regions=regions,
        evidence_bullets=evidence_bullets,
        retrieved_context=retrieved_context,
        reference_comparison=reference_comparison,
    )
    retrieved_block = [_serialize_retrieved_context_item(item) for item in (retrieved_context or [])]
    schema_block = build_schema_block()
    return {
        "block_a_detector_evidence": detector_block,
        "block_b_retrieved_support": retrieved_block,
        "block_c_output_schema": schema_block,
        "retrieval_query": build_retrieval_query(package),
        "support_status": _support_status(retrieved_block),
    }


def generate_operator_report(
    *,
    package: EvidencePackage,
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
    evidence_bullets: Sequence[str] | None = None,
    reference_comparison: str | None = None,
    provider: ExplanationProvider | None = None,
) -> Dict[str, Any]:
    context_pack = build_explainer_context_pack(
        package=package,
        retrieved_context=retrieved_context,
        regions=regions,
        evidence_bullets=evidence_bullets,
        reference_comparison=reference_comparison,
    )
    recommended_state = determine_explanation_state(
        package,
        context_pack["block_b_retrieved_support"],
    )

    provider_name = getattr(provider, "provider_name", "")
    model_name = getattr(provider, "model_name", "")
    provider_error = ""
    provider_available = provider is not None
    used_fallback = False
    raw_text = ""

    if provider is None:
        report = build_detector_only_report(
            package,
            recommended_state=recommended_state,
            retrieved_context=context_pack["block_b_retrieved_support"],
        )
        provider_name = "detector_fallback"
        used_fallback = True
    else:
        try:
            generated = provider.generate(
                context_pack["block_a_detector_evidence"],
                context_pack["block_b_retrieved_support"],
                context_pack["block_c_output_schema"],
            )
            raw_text = str(generated.get("text", ""))
            report = sanitize_operator_report(
                generated.get("structured_output", {}),
                retrieved_context=context_pack["block_b_retrieved_support"],
                recommended_state=recommended_state,
            )
            provider_name = str(generated.get("provider", provider_name or "provider"))
            model_name = str(generated.get("model", model_name))
        except Exception as exc:
            provider_error = str(exc)
            report = build_detector_only_report(
                package,
                recommended_state=recommended_state,
                retrieved_context=context_pack["block_b_retrieved_support"],
            )
            used_fallback = True

    return {
        "provider": provider_name,
        "model": model_name,
        "provider_available": provider_available,
        "provider_error": provider_error,
        "used_fallback": used_fallback,
        "text": raw_text,
        "structured_output": report.to_dict(),
        "explanation_state": report.explanation_state,
        "markdown": render_operator_report_markdown(report),
        "context_pack": context_pack,
    }


def sanitize_operator_report(
    payload: Mapping[str, Any],
    *,
    retrieved_context: Sequence[Mapping[str, Any]],
    recommended_state: str,
) -> OperatorReport:
    allowed_context = {
        (str(item.get("title", "")), str(item.get("source", ""))): item
        for item in retrieved_context
    }
    citations: list[SupportingCitation] = []
    for raw_citation in payload.get("supporting_citations", [])[:3]:
        if not isinstance(raw_citation, Mapping):
            continue
        key = (str(raw_citation.get("title", "")), str(raw_citation.get("source", "")))
        matched = allowed_context.get(key)
        if matched is None:
            continue
        citations.append(
            SupportingCitation(
                title=key[0],
                source=key[1],
                snippet=str(matched.get("snippet", "")),
                relevance_reason=str(raw_citation.get("relevance_reason", "") or matched.get("relevance_reason", "")),
            )
        )

    explanation_state = str(payload.get("explanation_state", recommended_state) or recommended_state).lower()
    if explanation_state not in {"grounded", "cautious", "abstained"}:
        explanation_state = recommended_state

    support_status = _support_status(retrieved_context)
    if not citations:
        explanation_state = _downgrade_explanation_state(explanation_state, recommended_state)
    elif support_status != "strong" and explanation_state == "grounded":
        explanation_state = _downgrade_explanation_state(explanation_state, recommended_state)

    return OperatorReport(
        explanation_state=explanation_state,
        likely_cause=str(payload.get("likely_cause", "Likely cause unavailable.")),
        why_flagged=str(payload.get("why_flagged", "The detector flagged a structured anomaly pattern.")),
        recommended_action=str(payload.get("recommended_action", "Manual inspection is recommended.")),
        operator_summary=str(payload.get("operator_summary", "An anomaly was detected and should be reviewed.")),
        supporting_citations=citations,
    )


def build_detector_only_report(
    package: EvidencePackage,
    *,
    recommended_state: str,
    retrieved_context: Sequence[Mapping[str, Any]] | None = None,
) -> OperatorReport:
    dominant_modalities = ", ".join(package.retrieval_features.dominant_modalities[:2]) or "detector evidence"
    gate_mode = str(package.provenance.get("internal_defect_gate_mode", "")).strip()
    retrieved_items = [_serialize_retrieved_context_item(item) for item in (retrieved_context or [])]
    supporting_citations = _fallback_supporting_citations(retrieved_items)
    likely_cause = (
        "Likely cause is uncertain from detector evidence alone."
        if recommended_state == "abstained"
        else f"Most likely anomaly profile is supported by {dominant_modalities}."
    )
    if gate_mode == "cross_inconsistency":
        likely_cause = "Cross-modal inconsistency suggests an internal defect profile."
    elif gate_mode == "thermal_hotspot":
        likely_cause = "Thermal hotspot evidence suggests an internally driven heat anomaly."
    elif supporting_citations:
        likely_cause = (
            f"Detector evidence aligns most closely with the trusted reference "
            f"'{supporting_citations[0].title}'."
        )

    why_flagged = (
        f"MulSenDiff-X flagged this sample with score {package.raw_score:.4f}, "
        f"severity {package.severity_0_100:.1f}/100, and affected area {package.affected_area_pct:.2f}%."
    )
    recommended_action = (
        "Inspect the dominant region first, compare the indicated sensor cues, "
        "and verify whether the observed pattern matches the suspected defect mechanism."
    )
    if supporting_citations:
        recommended_action = (
            f"Prioritize the checks described in '{supporting_citations[0].title}' and confirm whether "
            "the observed detector cues match the referenced defect signature before deciding on rework."
        )
    operator_summary = (
        f"{package.category} / {package.defect_label}: {package.status.lower()} with "
        f"{package.confidence_0_100:.1f}/100 confidence."
    )
    if supporting_citations:
        operator_summary += " Trusted reference support is available for this inspection."
    return OperatorReport(
        explanation_state=recommended_state,
        likely_cause=likely_cause,
        why_flagged=why_flagged,
        recommended_action=recommended_action,
        operator_summary=operator_summary,
        supporting_citations=supporting_citations,
    )


def render_operator_report_markdown(report: OperatorReport) -> str:
    lines = [
        "## Likely Cause",
        report.likely_cause,
        "",
        "## Why Flagged",
        report.why_flagged,
        "",
        "## Recommended Action",
        report.recommended_action,
        "",
        "## Operator Summary",
        report.operator_summary,
        "",
        "## Supporting Citations",
    ]
    if report.supporting_citations:
        for citation in report.supporting_citations:
            lines.append(
                f"- {citation.title} ({citation.source}): {citation.snippet} [{citation.relevance_reason}]"
            )
    else:
        lines.append("- No external support cited.")
    return "\n".join(lines)


def save_explanation_bundle(
    *,
    output_path: Path | str,
    package: EvidencePackage,
    generation: Mapping[str, Any],
    source_eval_run: Path | str | None = None,
    source_evidence_path: Path | str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_eval_run": str(source_eval_run or ""),
        "source_evidence_path": str(source_evidence_path or ""),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "provider": str(generation.get("provider", "")),
        "model": str(generation.get("model", "")),
        "provider_available": bool(generation.get("provider_available", False)),
        "provider_error": str(generation.get("provider_error", "")),
        "used_fallback": bool(generation.get("used_fallback", False)),
        "evidence_package": package.to_dict(),
        "retrieval_query": generation.get("context_pack", {}).get("retrieval_query", {}),
        "retrieved_context": generation.get("context_pack", {}).get("block_b_retrieved_support", []),
        "context_pack": generation.get("context_pack", {}),
        "generated_text": str(generation.get("text", "")),
        "structured_output": dict(generation.get("structured_output", {})),
        "explanation_state": str(generation.get("explanation_state", "")),
        "rendered_markdown": str(generation.get("markdown", "")),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def determine_explanation_state(
    package: EvidencePackage,
    retrieved_context: Sequence[Mapping[str, Any] | RetrievedContextItem] | None = None,
) -> str:
    mixed = any(
        "mixed" in str(item).lower() or "cautious" in str(item).lower()
        for item in [*package.cross_modal_support, *package.confidence_notes]
    )
    low_confidence = float(package.confidence_0_100) < 25.0
    moderate_confidence = float(package.confidence_0_100) < 60.0
    support_status = _support_status(retrieved_context or [])
    if low_confidence or (mixed and moderate_confidence):
        return "abstained"
    if support_status == "none" or moderate_confidence or mixed:
        return "cautious"
    if support_status == "weak":
        return "cautious"
    return "grounded"


def _serialize_retrieved_context_item(item: RetrievedContextItem | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(item, RetrievedContextItem):
        return item.to_dict()
    return {
        "title": str(item.get("title", "")),
        "snippet": str(item.get("snippet", "")),
        "source": str(item.get("source", "")),
        "score": float(item.get("score", 0.0)),
        "relevance_reason": str(item.get("relevance_reason", "")),
        "kind": str(item.get("kind", "trusted_corpus_chunk")),
        "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), Mapping) else {},
    }


def _serialize_regions(
    regions: Sequence[Mapping[str, Any]],
    package: EvidencePackage,
) -> list[Dict[str, Any]]:
    serialized: list[Dict[str, Any]] = []
    for index, region in enumerate(regions, start=1):
        serialized.append(
            {
                "rank": index,
                "label": str(region.get("label", "")),
                "score": round(float(region.get("score", 0.0)), 6),
                "bbox": list(region.get("bbox", [])),
                "centroid_xy": list(region.get("centroid_xy", [])),
                "area_percent": round(float(region.get("area_percent", 0.0)), 3),
                "pattern": str(region.get("pattern", "")),
                "summary": str(region.get("summary", "")),
            }
        )
    if not serialized and package.top_regions:
        top_region = package.top_regions[0]
        serialized.append(
            {
                "rank": 1,
                "label": "dominant region",
                "score": round(float(top_region.get("peak_score", package.peak_anomaly_0_100 / 100.0)), 6),
                "bbox": list(top_region.get("bbox_xyxy", [])),
                "centroid_xy": list(top_region.get("centroid_xy", [])),
                "area_percent": round(100.0 * float(top_region.get("area_fraction", 0.0)), 3),
                "pattern": "fallback",
                "summary": "Fallback dominant region from detector evidence.",
            }
        )
    return serialized


def _default_evidence_bullets(package: EvidencePackage) -> list[str]:
    return [
        *package.rgb_observations,
        *package.thermal_observations,
        *package.geometric_observations,
        *package.cross_modal_support,
    ]


def _uncertainty_signals(package: EvidencePackage) -> list[str]:
    signals: list[str] = []
    for note in package.confidence_notes:
        lowered = str(note).lower()
        if any(token in lowered for token in ("mixed", "limited", "cautious", "small", "boundary")):
            signals.append(str(note))
    if not signals and float(package.confidence_0_100) < 50.0:
        signals.append("Confidence is moderate or low, so the likely cause should be treated cautiously.")
    return signals


def _reference_comparison_from_context(context_payload: Sequence[Mapping[str, Any]]) -> str | None:
    for item in context_payload:
        if str(item.get("kind", "")) == "normal_profile":
            return str(item.get("snippet", "")).strip() or None
    return None


def _support_status(retrieved_context: Sequence[Mapping[str, Any] | RetrievedContextItem]) -> str:
    if not retrieved_context:
        return "none"
    top = _serialize_retrieved_context_item(retrieved_context[0])
    top_score = float(top.get("score", 0.0))
    category_score = float(top.get("metadata", {}).get("category_score", 0.0))
    if top_score >= 0.75 and category_score >= 1.0:
        return "strong"
    return "weak"


def _downgrade_explanation_state(explanation_state: str, recommended_state: str) -> str:
    if recommended_state == "abstained":
        return "abstained"
    if explanation_state == "grounded":
        return "cautious"
    return recommended_state if recommended_state in {"cautious", "abstained"} else explanation_state


def _fallback_supporting_citations(
    retrieved_context: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> list[SupportingCitation]:
    citations: list[SupportingCitation] = []
    for item in retrieved_context[: max(limit, 0) or 3]:
        citations.append(
            SupportingCitation(
                title=str(item.get("title", "")),
                source=str(item.get("source", "")),
                snippet=str(item.get("snippet", "")),
                relevance_reason=str(item.get("relevance_reason", "")),
            )
        )
    return citations


def _parse_structured_json_response(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1]
    candidate = candidate.strip()
    if candidate.endswith("```"):
        candidate = candidate[:-3].strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini did not return a JSON object.")
    return parsed
