from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.category_policies import ALL_CATEGORIES
from src.explainer.evidence_builder import EvidencePackage, RetrievalFeatures
from src.explainer.llm_pipeline import (
    GeminiExplanationProvider,
    build_detector_only_report,
    determine_explanation_state,
    generate_operator_report,
    load_gemini_provider_config,
    render_operator_report_markdown,
    save_explanation_bundle,
)
from src.explainer.retriever import RetrievedContextItem, retrieve_context_for_evidence
from src.utils.logger import write_history_csv, write_json


EXPLANATION_MODES = ("retrieval_only", "generator_only", "full")


@dataclass(frozen=True)
class ExplanationAuditCase:
    case_id: str
    case_group: str
    category: str
    defect_label: str
    sample_id: str
    sample_name: str
    is_anomalous: bool
    status: str
    raw_score: float
    confidence_0_100: float
    severity_0_100: float
    package_path: str
    report_path: str
    selection_reason: str
    qualitative_anchor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_evidence_package(path: Path | str) -> EvidencePackage:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    retrieval_features_payload = payload.get("retrieval_features", {})
    retrieval_features = RetrievalFeatures(
        category=str(retrieval_features_payload.get("category", payload.get("category", ""))),
        defect_label=str(retrieval_features_payload.get("defect_label", payload.get("defect_label", ""))),
        confidence_band=str(retrieval_features_payload.get("confidence_band", "")),
        severity_band=str(retrieval_features_payload.get("severity_band", "")),
        distribution=str(retrieval_features_payload.get("distribution", "")),
        dominant_modalities=[str(item) for item in retrieval_features_payload.get("dominant_modalities", [])],
        sensor_profile_tags=[str(item) for item in retrieval_features_payload.get("sensor_profile_tags", [])],
        defect_profile_tags=[str(item) for item in retrieval_features_payload.get("defect_profile_tags", [])],
        query_text=str(retrieval_features_payload.get("query_text", "")),
    )
    return EvidencePackage(
        category=str(payload.get("category", "")),
        split=str(payload.get("split", "")),
        defect_label=str(payload.get("defect_label", "")),
        sample_id=str(payload.get("sample_id", "")),
        is_anomalous=bool(payload.get("is_anomalous", False)),
        score_mode=str(payload.get("score_mode", "")),
        score_label=str(payload.get("score_label", "")),
        raw_score=float(payload.get("raw_score", 0.0)),
        severity_0_100=float(payload.get("severity_0_100", 0.0)),
        confidence_0_100=float(payload.get("confidence_0_100", 0.0)),
        status=str(payload.get("status", "")),
        affected_area_pct=float(payload.get("affected_area_pct", 0.0)),
        peak_anomaly_0_100=float(payload.get("peak_anomaly_0_100", 0.0)),
        top_regions=[dict(item) for item in payload.get("top_regions", [])],
        rgb_observations=[str(item) for item in payload.get("rgb_observations", [])],
        thermal_observations=[str(item) for item in payload.get("thermal_observations", [])],
        geometric_observations=[str(item) for item in payload.get("geometric_observations", [])],
        cross_modal_support=[str(item) for item in payload.get("cross_modal_support", [])],
        evidence_breakdown={str(key): float(value) for key, value in payload.get("evidence_breakdown", {}).items()},
        confidence_notes=[str(item) for item in payload.get("confidence_notes", [])],
        global_descriptor_score=float(payload.get("global_descriptor_score", 0.0)),
        retrieval_features=retrieval_features,
        retrieved_context=[dict(item) for item in payload.get("retrieved_context", [])],
        provenance=dict(payload.get("provenance", {})),
        source_paths={str(key): str(value) for key, value in payload.get("source_paths", {}).items()},
    )


def build_explanation_audit_manifest(
    *,
    eval_run_root: Path | str,
    output_path: Path | str,
) -> List[ExplanationAuditCase]:
    eval_root = Path(eval_run_root)
    evidence_index = _load_evidence_index(eval_root)
    selected_keys: set[tuple[str, str, str]] = set()
    cases: List[ExplanationAuditCase] = []

    anomalous_rows = [row for row in evidence_index if bool(row.get("is_anomalous", False))]
    good_rows = [row for row in evidence_index if not bool(row.get("is_anomalous", False))]

    for category in ALL_CATEGORIES:
        category_rows = [row for row in anomalous_rows if str(row.get("category", "")) == category]
        if not category_rows:
            continue
        ranked = sorted(
            category_rows,
            key=lambda row: (
                str(row.get("status", "")).lower() != "normal",
                float(row.get("confidence_0_100", 0.0)),
                float(row.get("raw_score", 0.0)),
            ),
            reverse=True,
        )
        chosen = ranked[0]
        selected_keys.add(_row_key(chosen))
        cases.append(
            _build_case(
                chosen,
                case_index=len(cases) + 1,
                case_group="representative_anomaly",
                selection_reason="Highest-confidence anomalous case selected for the category anchor.",
            )
        )

    hard_good_rows = _take_diverse_rows(
        [
            row
            for row in good_rows
            if _row_key(row) not in selected_keys
        ],
        count=2,
        sort_key=lambda row: (
            str(row.get("status", "")).lower() != "normal",
            float(row.get("raw_score", 0.0)),
            float(row.get("confidence_0_100", 0.0)),
        ),
        reverse=True,
    )
    for row in hard_good_rows:
        selected_keys.add(_row_key(row))
        cases.append(
            _build_case(
                row,
                case_index=len(cases) + 1,
                case_group="hard_good",
                selection_reason="Good sample with the strongest suspicious detector evidence.",
            )
        )

    hard_anomaly_rows = _take_diverse_rows(
        [
            row
            for row in anomalous_rows
            if _row_key(row) not in selected_keys
        ],
        count=3,
        sort_key=lambda row: (
            float(row.get("confidence_0_100", 0.0)),
            float(row.get("severity_0_100", 0.0)),
            float(row.get("raw_score", 0.0)),
        ),
        reverse=False,
    )
    for row in hard_anomaly_rows:
        selected_keys.add(_row_key(row))
        cases.append(
            _build_case(
                row,
                case_index=len(cases) + 1,
                case_group="hard_anomaly",
                selection_reason="Anomalous case with weaker confidence or mixed detector evidence.",
            )
        )

    if cases:
        anchor_index = max(
            range(len(cases)),
            key=lambda idx: (
                cases[idx].case_group == "representative_anomaly",
                cases[idx].confidence_0_100,
                cases[idx].raw_score,
            ),
        )
        anchor = cases[anchor_index]
        cases[anchor_index] = ExplanationAuditCase(**{**anchor.to_dict(), "qualitative_anchor": True})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_history_csv(output, [case.to_dict() for case in cases])
    return cases


def run_explanation_ablation(
    *,
    audit_manifest_csv: Path | str,
    output_dir: Path | str,
    knowledge_base_root: Path | str | None = None,
    retrieval_top_k: int = 3,
    modes: Sequence[str] | None = None,
    require_generator: bool = True,
) -> Dict[str, Any]:
    selected_modes = [str(mode).strip().lower() for mode in (modes or EXPLANATION_MODES) if str(mode).strip()]
    for mode in selected_modes:
        if mode not in EXPLANATION_MODES:
            raise KeyError(f"Unknown explanation ablation mode: {mode}")

    cases = _read_audit_manifest(audit_manifest_csv)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    gemini_config = load_gemini_provider_config()
    if require_generator and any(mode in {"generator_only", "full"} for mode in selected_modes) and not gemini_config.enabled:
        raise RuntimeError(
            "Gemini is not configured. Add config/gemini.local.json or environment keys before running generator modes."
        )
    provider = GeminiExplanationProvider(gemini_config) if gemini_config.enabled else None

    mode_rows: List[Dict[str, Any]] = []
    blinded_rows: List[Dict[str, Any]] = []
    mode_key_rows: List[Dict[str, Any]] = []

    for case in cases:
        package = load_evidence_package(case.package_path)
        retrieved_context = retrieve_context_for_evidence(
            package,
            knowledge_base_root=knowledge_base_root,
            top_k=retrieval_top_k,
        )
        for mode in selected_modes:
            mode_dir = output_root / "cases" / case.case_id / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            mode_retrieved_context = retrieved_context if mode in {"retrieval_only", "full"} else []
            mode_provider = None if mode == "retrieval_only" else provider
            generation = generate_operator_report(
                package=package,
                retrieved_context=mode_retrieved_context,
                provider=mode_provider,
            )
            bundle_path = mode_dir / "bundle.json"
            markdown_path = mode_dir / "report.md"
            save_explanation_bundle(
                output_path=bundle_path,
                package=package,
                generation=generation,
                source_eval_run=Path(case.package_path).parents[4] if len(Path(case.package_path).parents) >= 5 else None,
                source_evidence_path=case.package_path,
            )
            markdown_path.write_text(str(generation.get("markdown", "")), encoding="utf-8")

            citation_count = len(generation.get("structured_output", {}).get("supporting_citations", []))
            retrieved_count = len(mode_retrieved_context)
            artifact_id = f"{case.case_id}__{mode}"
            artifact_markdown_path = output_root / "blinded_artifacts" / f"{artifact_id}.md"
            artifact_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_markdown_path.write_text(str(generation.get("markdown", "")), encoding="utf-8")

            mode_rows.append(
                {
                    "artifact_id": artifact_id,
                    "case_id": case.case_id,
                    "case_group": case.case_group,
                    "category": case.category,
                    "defect_label": case.defect_label,
                    "sample_id": case.sample_id,
                    "mode": mode,
                    "provider": str(generation.get("provider", "")),
                    "used_fallback": bool(generation.get("used_fallback", False)),
                    "explanation_state": str(generation.get("explanation_state", "")),
                    "retrieved_context_count": retrieved_count,
                    "citation_count": citation_count,
                    "no_citation": int(citation_count <= 0),
                    "citation_validity_auto": int(
                        _citations_are_grounded(
                            generation.get("structured_output", {}).get("supporting_citations", []),
                            generation.get("context_pack", {}).get("block_b_retrieved_support", []),
                        )
                    ),
                    "bundle_path": str(bundle_path),
                    "markdown_path": str(markdown_path),
                    "blinded_artifact_path": str(artifact_markdown_path),
                    "qualitative_anchor": int(case.qualitative_anchor),
                }
            )
            blinded_rows.append(
                {
                    "artifact_id": artifact_id,
                    "case_group": case.case_group,
                    "category": case.category,
                    "defect_label": case.defect_label,
                    "sample_id": case.sample_id,
                    "artifact_path": str(artifact_markdown_path),
                }
            )
            mode_key_rows.append(
                {
                    "artifact_id": artifact_id,
                    "case_id": case.case_id,
                    "mode": mode,
                    "bundle_path": str(bundle_path),
                }
            )

    write_history_csv(output_root / "metrics" / "mode_outputs.csv", mode_rows)
    write_history_csv(output_root / "metrics" / "mode_summary_auto.csv", _summarize_mode_rows(mode_rows))
    write_history_csv(output_root / "ratings" / "mode_key.csv", mode_key_rows)
    write_history_csv(output_root / "ratings" / "blinded_manifest.csv", blinded_rows)
    write_history_csv(output_root / "ratings" / "rater1_template.csv", _build_rating_template(blinded_rows))
    write_history_csv(output_root / "ratings" / "rater2_template.csv", _build_rating_template(blinded_rows))
    _write_triptych(output_root=output_root, mode_rows=mode_rows)

    summary = {
        "cases": len(cases),
        "modes": selected_modes,
        "output_dir": str(output_root),
        "mode_outputs_csv": str(output_root / "metrics" / "mode_outputs.csv"),
        "mode_summary_auto_csv": str(output_root / "metrics" / "mode_summary_auto.csv"),
        "rater1_template_csv": str(output_root / "ratings" / "rater1_template.csv"),
        "rater2_template_csv": str(output_root / "ratings" / "rater2_template.csv"),
        "triptych_path": str(output_root / "qualitative_triptych.md"),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def summarize_explanation_ratings(
    *,
    output_dir: Path | str,
    rating_csv_paths: Sequence[Path | str],
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    mode_key = {
        str(row["artifact_id"]): dict(row)
        for row in _read_csv_rows(output_root / "ratings" / "mode_key.csv")
    }
    mode_rows = {str(row["artifact_id"]): dict(row) for row in _read_csv_rows(output_root / "metrics" / "mode_outputs.csv")}

    per_artifact: Dict[str, Dict[str, List[float]]] = {}
    agreement_pairs: Dict[str, Dict[str, List[float]]] = {}
    for rating_path in rating_csv_paths:
        for row in _read_csv_rows(rating_path):
            artifact_id = str(row.get("artifact_id", "")).strip()
            if artifact_id not in mode_key:
                continue
            bucket = per_artifact.setdefault(
                artifact_id,
                {
                    "grounding_correctness": [],
                    "evidence_coverage": [],
                    "citation_validity": [],
                    "operator_usefulness": [],
                },
            )
            for field in bucket:
                value = _coerce_optional_float(row.get(field, ""))
                if value is not None:
                    bucket[field].append(value)
                    agreement_pairs.setdefault(artifact_id, {}).setdefault(field, []).append(value)

    artifact_rows: List[Dict[str, Any]] = []
    for artifact_id, scores in sorted(per_artifact.items()):
        mode = str(mode_key[artifact_id]["mode"])
        auto_row = mode_rows.get(artifact_id, {})
        artifact_rows.append(
            {
                "artifact_id": artifact_id,
                "case_id": str(mode_key[artifact_id]["case_id"]),
                "mode": mode,
                "grounding_correctness_mean": _mean(scores["grounding_correctness"]),
                "evidence_coverage_mean": _mean(scores["evidence_coverage"]),
                "citation_validity_mean": _mean(scores["citation_validity"]),
                "operator_usefulness_mean": _mean(scores["operator_usefulness"]),
                "unsupported_claim": int(_mean(scores["grounding_correctness"]) < 0.5),
                "no_citation": int(int(auto_row.get("no_citation", 0)) > 0),
            }
        )

    mode_summary_rows: List[Dict[str, Any]] = []
    for mode in EXPLANATION_MODES:
        rows = [row for row in artifact_rows if str(row.get("mode")) == mode]
        if not rows:
            continue
        mode_summary_rows.append(
            {
                "mode": mode,
                "cases": len(rows),
                "grounding_correctness_mean": round(_mean([row["grounding_correctness_mean"] for row in rows]), 6),
                "evidence_coverage_mean": round(_mean([row["evidence_coverage_mean"] for row in rows]), 6),
                "citation_validity_mean": round(_mean([row["citation_validity_mean"] for row in rows]), 6),
                "operator_usefulness_mean": round(_mean([row["operator_usefulness_mean"] for row in rows]), 6),
                "unsupported_claim_rate": round(_mean([row["unsupported_claim"] for row in rows]), 6),
                "no_citation_rate": round(_mean([row["no_citation"] for row in rows]), 6),
            }
        )

    write_history_csv(output_root / "ratings" / "artifact_rating_summary.csv", artifact_rows)
    write_history_csv(output_root / "ratings" / "mode_rating_summary.csv", mode_summary_rows)
    summary = {
        "artifact_rating_summary_csv": str(output_root / "ratings" / "artifact_rating_summary.csv"),
        "mode_rating_summary_csv": str(output_root / "ratings" / "mode_rating_summary.csv"),
        "rating_csv_paths": [str(Path(path)) for path in rating_csv_paths],
    }
    write_json(output_root / "ratings" / "summary.json", summary)
    return summary


def _read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_evidence_index(eval_run_root: Path) -> List[Dict[str, Any]]:
    path = eval_run_root / "evidence" / "index.json"
    return [dict(row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("category", "")),
        str(row.get("defect_label", "")),
        str(row.get("sample_id", "")),
    )


def _build_case(
    row: Mapping[str, Any],
    *,
    case_index: int,
    case_group: str,
    selection_reason: str,
) -> ExplanationAuditCase:
    category = str(row.get("category", ""))
    defect_label = str(row.get("defect_label", ""))
    sample_id = str(row.get("sample_id", ""))
    case_id = f"case_{case_index:02d}__{category}__{defect_label}__{sample_id}"
    return ExplanationAuditCase(
        case_id=case_id,
        case_group=case_group,
        category=category,
        defect_label=defect_label,
        sample_id=sample_id,
        sample_name=str(row.get("sample_name", sample_id)),
        is_anomalous=bool(row.get("is_anomalous", False)),
        status=str(row.get("status", "")),
        raw_score=float(row.get("raw_score", 0.0)),
        confidence_0_100=float(row.get("confidence_0_100", 0.0)),
        severity_0_100=float(row.get("severity_0_100", 0.0)),
        package_path=str(row.get("package_path", "")),
        report_path=str(row.get("report_path", "")),
        selection_reason=selection_reason,
    )


def _take_diverse_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    sort_key,
    reverse: bool,
) -> List[Mapping[str, Any]]:
    ranked = sorted(rows, key=sort_key, reverse=reverse)
    selected: List[Mapping[str, Any]] = []
    seen_categories: set[str] = set()
    for row in ranked:
        category = str(row.get("category", ""))
        if category in seen_categories:
            continue
        selected.append(row)
        seen_categories.add(category)
        if len(selected) >= count:
            return selected
    for row in ranked:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _read_audit_manifest(path: Path | str) -> List[ExplanationAuditCase]:
    rows = _read_csv_rows(path)
    return [
        ExplanationAuditCase(
            case_id=str(row.get("case_id", "")),
            case_group=str(row.get("case_group", "")),
            category=str(row.get("category", "")),
            defect_label=str(row.get("defect_label", "")),
            sample_id=str(row.get("sample_id", "")),
            sample_name=str(row.get("sample_name", "")),
            is_anomalous=str(row.get("is_anomalous", "")).strip().lower() in {"1", "true", "yes"},
            status=str(row.get("status", "")),
            raw_score=float(row.get("raw_score", 0.0) or 0.0),
            confidence_0_100=float(row.get("confidence_0_100", 0.0) or 0.0),
            severity_0_100=float(row.get("severity_0_100", 0.0) or 0.0),
            package_path=str(row.get("package_path", "")),
            report_path=str(row.get("report_path", "")),
            selection_reason=str(row.get("selection_reason", "")),
            qualitative_anchor=str(row.get("qualitative_anchor", "")).strip().lower() in {"1", "true", "yes"},
        )
        for row in rows
    ]


def _citations_are_grounded(
    citations: Sequence[Mapping[str, Any]],
    retrieved_support: Sequence[Mapping[str, Any]],
) -> bool:
    allowed = {
        (str(item.get("title", "")), str(item.get("source", "")))
        for item in retrieved_support
    }
    for citation in citations:
        key = (str(citation.get("title", "")), str(citation.get("source", "")))
        if key not in allowed:
            return False
    return True


def _summarize_mode_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for mode in EXPLANATION_MODES:
        mode_rows = [row for row in rows if str(row.get("mode", "")) == mode]
        if not mode_rows:
            continue
        summary_rows.append(
            {
                "mode": mode,
                "cases": len(mode_rows),
                "citation_count_mean": round(_mean([float(row.get("citation_count", 0)) for row in mode_rows]), 6),
                "retrieved_context_count_mean": round(
                    _mean([float(row.get("retrieved_context_count", 0)) for row in mode_rows]),
                    6,
                ),
                "no_citation_rate": round(_mean([float(row.get("no_citation", 0)) for row in mode_rows]), 6),
                "citation_validity_auto_rate": round(
                    _mean([float(row.get("citation_validity_auto", 0)) for row in mode_rows]),
                    6,
                ),
                "fallback_rate": round(
                    _mean([1.0 if bool(row.get("used_fallback", False)) else 0.0 for row in mode_rows]),
                    6,
                ),
            }
        )
    return summary_rows


def _build_rating_template(blinded_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "artifact_id": str(row.get("artifact_id", "")),
            "case_group": str(row.get("case_group", "")),
            "category": str(row.get("category", "")),
            "defect_label": str(row.get("defect_label", "")),
            "sample_id": str(row.get("sample_id", "")),
            "grounding_correctness": "",
            "evidence_coverage": "",
            "citation_validity": "",
            "operator_usefulness": "",
            "notes": "",
        }
        for row in blinded_rows
    ]


def _write_triptych(*, output_root: Path, mode_rows: Sequence[Mapping[str, Any]]) -> None:
    anchor_rows = [row for row in mode_rows if int(row.get("qualitative_anchor", 0)) > 0]
    if not anchor_rows:
        return
    anchor_case_id = str(anchor_rows[0].get("case_id", ""))
    mode_order = {mode: index for index, mode in enumerate(EXPLANATION_MODES)}
    case_rows = sorted(
        [row for row in mode_rows if str(row.get("case_id", "")) == anchor_case_id],
        key=lambda row: mode_order.get(str(row.get("mode", "")), 99),
    )
    lines = [f"# Qualitative Triptych: {anchor_case_id}", ""]
    for row in case_rows:
        lines.append(f"## {row.get('mode', '')}")
        markdown_path = Path(str(row.get("markdown_path", "")))
        if markdown_path.exists():
            lines.append(markdown_path.read_text(encoding="utf-8").strip())
        lines.append("")
    (output_root / "qualitative_triptych.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / max(len(values), 1)
