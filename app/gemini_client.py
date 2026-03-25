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
    reference_comparison: str | None = None,
) -> Dict[str, Any]:
    status_label = "normal" if package.status.lower().startswith("normal") else "anomalous"
    payload: Dict[str, Any] = {
        "status": status_label,
        "anomaly_score": round(float(package.raw_score), 6),
        "severity": {
            "label": package.status,
            "value_0_100": round(float(package.severity_0_100), 3),
        },
        "affected_area_percent": round(float(package.affected_area_pct), 3),
        "regions": [
            {
                "label": str(region.get("label", "")),
                "score": round(float(region.get("score", 0.0)), 6),
                "bbox": list(region.get("bbox", [])),
            }
            for region in regions
        ],
        "evidence": [str(item) for item in evidence_bullets],
        "category": package.category,
    }
    if reference_comparison:
        payload["reference_comparison"] = str(reference_comparison)
    return payload


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
        "If the evidence is weak, state that the cause is only a likely hypothesis.",
        "",
        "Return valid JSON with exactly these keys:",
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
            "likely_cause": {"type": "STRING"},
            "why_flagged": {"type": "STRING"},
            "recommended_action": {"type": "STRING"},
            "operator_summary": {"type": "STRING"},
        },
        "required": [
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
