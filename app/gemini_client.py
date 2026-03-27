from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.explainer.evidence_builder import EvidencePackage
from src.explainer.retriever import RetrievedContextItem


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)


def load_gemini_config(env: Mapping[str, str] | None = None) -> GeminiConfig:
    resolved_env = dict(env or os.environ)
    api_key = str(
        resolved_env.get("GOOGLE_API_KEY")
        or resolved_env.get("GEMINI_API_KEY")
        or resolved_env.get("MULSENDIFFX_GEMINI_API_KEY")
        or ""
    ).strip()
    model = str(
        resolved_env.get("MULSENDIFFX_GEMINI_MODEL")
        or resolved_env.get("GEMINI_MODEL")
        or "gemini-2.5-flash"
    ).strip()
    return GeminiConfig(api_key=api_key, model=model)


def build_root_cause_evidence(
    *,
    package: EvidencePackage,
    regions: Sequence[Mapping[str, Any]],
    evidence_bullets: Sequence[str],
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
    reference_comparison: str | None = None,
) -> Dict[str, Any]:
    status_label = "normal" if package.status.lower().startswith("normal") else "anomalous"
    explanation_state = determine_explanation_state(package)
    context_payload = [_serialize_retrieved_context_item(item) for item in (retrieved_context or [])]
    localized_regions = _serialize_regions(regions, package)
    pattern_summary = _build_pattern_summary(package, localized_regions)
    payload: Dict[str, Any] = {
        "status": status_label,
        "anomaly_score": round(float(package.raw_score), 6),
        "severity": {
            "label": package.status,
            "value_0_100": round(float(package.severity_0_100), 3),
        },
        "confidence": {
            "value_0_100": round(float(package.confidence_0_100), 3),
            "evidence_support_score": round(_evidence_support_score(package), 3),
            "recommended_explanation_state": explanation_state,
        },
        "affected_area_percent": round(float(package.affected_area_pct), 3),
        "localized_regions": localized_regions,
        "anomaly_pattern": pattern_summary,
        "evidence": [str(item) for item in evidence_bullets],
        "confidence_notes": [str(item) for item in package.confidence_notes],
        "uncertainty_signals": _uncertainty_signals(package),
        "normal_reference_comparison": reference_comparison or _reference_comparison_from_context(context_payload),
        "category": package.category,
        "reference_examples": context_payload,
    }
    return payload


def determine_explanation_state(package: EvidencePackage) -> str:
    mixed = any(
        "mixed" in str(item).lower() or "cautious" in str(item).lower()
        for item in [*package.cross_modal_support, *package.confidence_notes]
    )
    low_confidence = float(package.confidence_0_100) < 25.0
    moderate_confidence = float(package.confidence_0_100) < 60.0
    if low_confidence or (mixed and moderate_confidence):
        return "abstained"
    if moderate_confidence or mixed:
        return "cautious"
    return "grounded"


def generate_root_cause_explanation(
    evidence: Dict[str, Any],
    *,
    config: GeminiConfig,
    retrieved_context: Sequence[RetrievedContextItem | Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    if not config.enabled:
        raise RuntimeError("Gemini is not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY and rerun the inspection.")

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "The `google-genai` package is not installed. Run `pip install google-genai` in your app environment."
        ) from exc

    context_payload = [_serialize_retrieved_context_item(item) for item in (retrieved_context or [])]
    prompt_sections = [
        "You are an industrial anomaly inspection assistant.",
        "Use only the evidence provided.",
        "Do not overclaim or invent missing causes.",
        "Speak in practical operator-friendly language.",
        "If the evidence is weak or mixed, abstain from a specific root cause and say the likely cause is uncertain.",
        "Treat the detector as the source of truth and Gemini as the explanation layer only.",
        "",
        "Return valid JSON with exactly these keys:",
        "- explanation_state",
        "- likely_cause",
        "- why_flagged",
        "- recommended_action",
        "- operator_summary",
        "",
        "Detection evidence:",
        json.dumps(evidence, indent=2, sort_keys=True),
    ]
    if context_payload:
        prompt_sections.extend(
            [
                "",
                "Retrieved context:",
                json.dumps(context_payload, indent=2, sort_keys=True),
            ]
        )
    prompt = "\n".join(prompt_sections)

    response_schema = {
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
        },
        "required": [
            "explanation_state",
            "likely_cause",
            "why_flagged",
            "recommended_action",
            "operator_summary",
        ],
    }

    client = genai.Client(api_key=config.api_key)
    response = client.models.generate_content(
        model=config.model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": 0.2,
        },
    )
    text = str(getattr(response, "text", "") or "").strip()
    structured_output = _parse_structured_json_response(text)
    explanation_state = str(
        structured_output.get("explanation_state")
        or evidence.get("confidence", {}).get("recommended_explanation_state", "cautious")
    )
    structured_output = {
        "likely_cause": str(structured_output.get("likely_cause", "Likely cause unavailable.")),
        "why_flagged": str(structured_output.get("why_flagged", "The detector flagged a structured anomaly pattern.")),
        "recommended_action": str(
            structured_output.get("recommended_action", "Manual inspection is recommended.")
        ),
        "operator_summary": str(
            structured_output.get("operator_summary", "An anomaly was detected and should be reviewed.")
        ),
    }

    return {
        "provider": "gemini",
        "model": config.model,
        "text": text,
        "structured_output": structured_output,
        "explanation_state": explanation_state,
        "markdown": render_root_cause_markdown(structured_output),
    }


def render_root_cause_markdown(explanation: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "## Likely Cause",
            str(explanation.get("likely_cause", "")),
            "",
            "## Why Flagged",
            str(explanation.get("why_flagged", "")),
            "",
            "## Recommended Action",
            str(explanation.get("recommended_action", "")),
            "",
            "## Operator Summary",
            str(explanation.get("operator_summary", "")),
        ]
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
    output_root = Path(output_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_root = output_root / f"{timestamp}_{sample_slug}"
    session_root.mkdir(parents=True, exist_ok=True)

    markdown_path = session_root / "gemini_explanation.md"
    markdown_path.write_text(str(generation["markdown"]), encoding="utf-8")

    json_path = session_root / "gemini_explanation.json"
    payload = {
        "source_eval_run": str(source_eval_run),
        "source_evidence_path": str(source_evidence_path),
        "provider": str(generation.get("provider", "gemini")),
        "model": str(generation.get("model", "")),
        "timestamp_utc": timestamp,
        "evidence_package": package.to_dict(),
        "retrieved_context": [_serialize_retrieved_context_item(item) for item in retrieved_context],
        "evidence_payload": prompt_payload,
        "prompt_payload": prompt_payload,
        "generated_markdown": str(generation["markdown"]),
        "generated_text": str(generation.get("text", generation["markdown"])),
        "structured_output": generation.get("structured_output", {}),
        "explanation_state": str(generation.get("explanation_state", "")),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "session_root": str(session_root),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
    }


def _serialize_retrieved_context_item(item: RetrievedContextItem | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(item, RetrievedContextItem):
        return item.to_dict()
    return {
        "title": str(item.get("title", "")),
        "snippet": str(item.get("snippet", "")),
        "source": str(item.get("source", "")),
        "score": float(item.get("score", 0.0)),
        "kind": str(item.get("kind", "document")),
        "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), Mapping) else {},
    }


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


def _build_pattern_summary(package: EvidencePackage, localized_regions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    dominant_sources = sorted(
        package.evidence_breakdown.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    top_sources = [key.replace("_pct", "") for key, value in dominant_sources if float(value) > 0.0][:3]
    dominant_region = localized_regions[0] if localized_regions else {}
    return {
        "peak_anomaly_0_100": round(float(package.peak_anomaly_0_100), 3),
        "distribution": _distribution_label(package.affected_area_pct),
        "dominant_region_summary": str(dominant_region.get("summary", "")),
        "dominant_evidence_sources": top_sources,
        "score_label": package.score_label,
    }


def _distribution_label(affected_area_pct: float) -> str:
    if affected_area_pct >= 20.0:
        return "broad"
    if affected_area_pct >= 7.5:
        return "moderately concentrated"
    return "compact"


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


def _evidence_support_score(package: EvidencePackage) -> float:
    mixed_penalty = 0.25 if any("mixed" in str(item).lower() for item in package.cross_modal_support) else 0.0
    local_support = (
        float(package.evidence_breakdown.get("score_basis_pct", 0.0))
        + float(package.evidence_breakdown.get("thermal_pct", 0.0))
        + float(package.evidence_breakdown.get("geometric_pct", 0.0))
        + float(package.evidence_breakdown.get("crossmodal_pct", 0.0))
    ) / 100.0
    confidence_component = float(package.confidence_0_100) / 100.0
    score = 0.6 * confidence_component + 0.4 * local_support - mixed_penalty
    return max(0.0, min(1.0, score))
