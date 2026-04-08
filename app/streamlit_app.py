from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.dashboard_data import (
    AvailableRun,
    default_app_sessions_root,
    find_available_evaluation_runs,
    find_known_processed_samples,
    load_run_bundle,
)
from app.demo_models import (
    EvidenceSummary,
    ExplanationBundle,
    InspectionBundle,
    InspectionVisuals,
    RetrievalStatus,
    retrieval_items_to_dicts,
)
from app.gemini_client import (
    build_root_cause_evidence,
    generate_root_cause_explanation,
    load_gemini_config,
    save_gemini_generation,
)
from app.inference_runtime import SharedInferenceSession, load_shared_inference_session, run_known_sample_inference
from src.data_loader import ProcessedSampleRecord
from src.explainer.evidence_builder import EvidencePackage
from src.explainer.retriever import load_trusted_corpus, resolve_corpus_index_path


st.set_page_config(
    page_title="MulSenDiff-X Inspector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DISPLAY_PANEL_SIZE = 400
REFERENCE_PANEL_SIZE = 150
DEFAULT_DEMO_CATEGORY = "zipper"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
                color: #0f172a;
            }
            .block-container {
                max-width: 1320px;
                padding-top: 0.8rem;
                padding-bottom: 0.7rem;
            }
            h1, h2, h3, h4, p, label, span {
                color: #0f172a;
            }
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
                color: #0f172a !important;
            }
            [data-testid="stCaptionContainer"] {
                color: #475569 !important;
            }
            [data-testid="stSelectbox"] label,
            [data-testid="stButton"] label {
                color: #0f172a !important;
            }
            .stButton > button {
                border-radius: 14px;
                height: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
                border: none;
                color: white;
            }
            .stDownloadButton > button {
                border-radius: 14px;
                height: 2.8rem;
                font-weight: 700;
            }
            .stSelectbox [data-baseweb="select"] {
                border-radius: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_eval_bundle_cached(run_root_str: str) -> Dict[str, Any]:
    return load_run_bundle(run_root_str)


@st.cache_data(show_spinner=False)
def load_retrieval_status_cached(retrieval_root_str: str) -> Dict[str, Any]:
    retrieval_root = Path(retrieval_root_str)
    index_path = resolve_corpus_index_path(retrieval_root)
    if index_path is None:
        return {"configured": False, "index_path": "", "chunk_count": 0}
    chunks = load_trusted_corpus(index_path)
    return {"configured": True, "index_path": str(index_path), "chunk_count": len(chunks)}


@st.cache_resource(show_spinner=False)
def load_inference_session_cached(eval_run_root_str: str) -> SharedInferenceSession:
    return load_shared_inference_session(repo_root=REPO_ROOT, eval_run_root=eval_run_root_str)


@st.cache_data(show_spinner=False)
def load_known_records_cached(
    categories: tuple[str, ...],
    processed_root_str: str = "",
) -> List[ProcessedSampleRecord]:
    return find_known_processed_samples(
        REPO_ROOT,
        categories=categories,
        processed_root=processed_root_str or None,
    )


def resolve_primary_demo_run(evaluation_runs: Sequence[AvailableRun]) -> AvailableRun:
    ccdd_runs = [
        run
        for run in evaluation_runs
        if "ccdd" in run.name.lower() or "__ccdd__" in run.name.lower()
    ]
    ranked = sorted(ccdd_runs or list(evaluation_runs), key=lambda run: run.name)
    return ranked[-1]


def build_curated_demo_records(
    *,
    preview_selection: Mapping[str, Any],
    known_records: Sequence[ProcessedSampleRecord],
) -> List[ProcessedSampleRecord]:
    by_name = {record.sample_name: record for record in known_records}
    curated: List[ProcessedSampleRecord] = []
    for item in preview_selection.get("selected_samples", []):
        sample_name = str(item.get("sample_name", "")).strip()
        category = str(item.get("category", "")).strip()
        defect_label = str(item.get("defect_label", "")).strip()
        sample_id = str(item.get("sample_id", "")).strip()
        record = by_name.get(sample_name)
        if record is None:
            record = next(
                (
                    candidate
                    for candidate in known_records
                    if candidate.category == category
                    and candidate.defect_label == defect_label
                    and candidate.sample_id == sample_id
                ),
                None,
            )
        if record is not None:
            curated.append(record)
    if curated:
        return curated
    anomaly_records = [record for record in known_records if record.is_anomalous]
    return anomaly_records[:15] if anomaly_records else list(known_records[:15])


def categories_in_demo(records: Sequence[ProcessedSampleRecord]) -> List[str]:
    categories: List[str] = []
    seen = set()
    for record in records:
        if record.category in seen:
            continue
        seen.add(record.category)
        categories.append(record.category)
    return categories


def choose_default_category(categories: Sequence[str]) -> str:
    if DEFAULT_DEMO_CATEGORY in categories:
        return DEFAULT_DEMO_CATEGORY
    return categories[0] if categories else ""


def records_for_category(
    records: Sequence[ProcessedSampleRecord],
    category: str,
) -> List[ProcessedSampleRecord]:
    return [record for record in records if record.category == category]


def format_sample_option(record: ProcessedSampleRecord) -> str:
    state = "anomaly" if record.is_anomalous else "normal"
    return f"{record.defect_label} / {record.sample_id} / {state}"


def tensor_to_rgb_array(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def blend_overlay(rgb: torch.Tensor, anomaly_map: torch.Tensor) -> np.ndarray:
    rgb_array = tensor_to_rgb_array(rgb).astype(np.float32) / 255.0
    score = anomaly_map.detach().cpu().numpy().astype(np.float32)
    if score.ndim == 3:
        score = score.squeeze(0)
    score = np.clip(score, 0.0, 1.0)
    heatmap = np.zeros_like(rgb_array)
    heatmap[..., 0] = score
    heatmap[..., 1] = np.clip(score * 0.45, 0.0, 1.0)
    overlay = rgb_array * (1.0 - (0.45 * score[..., None])) + heatmap * (0.60 * score[..., None])
    return (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)


def _display_background() -> tuple[int, int, int]:
    return (248, 250, 252)


def upscale_for_display(
    image: Image.Image | np.ndarray,
    *,
    target_size: int,
    sharpen: bool = True,
) -> Image.Image:
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil_image = pil_image.convert("RGB")
    contained = pil_image.copy()
    contained.thumbnail((target_size, target_size), resample=Image.Resampling.LANCZOS)
    if sharpen:
        contained = contained.filter(ImageFilter.UnsharpMask(radius=1.1, percent=130, threshold=2))
    canvas = Image.new("RGB", (target_size, target_size), _display_background())
    left = max((target_size - contained.width) // 2, 0)
    top = max((target_size - contained.height) // 2, 0)
    canvas.paste(contained, (left, top))
    return canvas


def find_reference_record(records: Sequence[ProcessedSampleRecord], category: str) -> ProcessedSampleRecord | None:
    good_records = [
        record
        for record in records
        if record.category == category and record.split == "train" and record.defect_label == "good"
    ]
    return good_records[0] if good_records else None


def load_reference_image(record: ProcessedSampleRecord | None) -> Image.Image | None:
    if record is None:
        return None
    candidates = [Path(record.sample_dir) / "rgb.png", Path(record.rgb_source_path)]
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        if resolved.exists():
            return Image.open(resolved).convert("RGB")
    return None


def region_location_label(centroid_xy: tuple[float, float], image_shape: tuple[int, int]) -> str:
    height, width = image_shape
    x, y = centroid_xy
    horizontal = "left" if x < width * 0.33 else "right" if x > width * 0.66 else "center"
    vertical = "upper" if y < height * 0.33 else "lower" if y > height * 0.66 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "central region"
    if horizontal == "center":
        return f"{vertical} region"
    if vertical == "middle":
        return f"{horizontal} side"
    return f"{vertical}-{horizontal} region"


def region_intensity_label(score: float) -> str:
    if score >= 0.80:
        return "high anomaly"
    if score >= 0.50:
        return "medium anomaly"
    return "low anomaly"


def extract_detected_regions(
    *,
    predicted_mask: torch.Tensor,
    anomaly_map: torch.Tensor,
    object_mask: torch.Tensor | None = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    mask = predicted_mask.detach().cpu().numpy().astype(bool)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    score_map = anomaly_map.detach().cpu().numpy().astype(np.float32)
    if score_map.ndim == 3:
        score_map = score_map.squeeze(0)
    object_region = None
    if object_mask is not None:
        object_region = object_mask.detach().cpu().numpy().astype(bool)
        if object_region.ndim == 3:
            object_region = object_region.squeeze(0)
    object_pixels = int(object_region.sum()) if object_region is not None else int(mask.size)
    object_pixels = max(object_pixels, 1)

    labeled, count = ndimage.label(mask)
    regions: List[Dict[str, Any]] = []
    for region_index in range(1, count + 1):
        region_mask = labeled == region_index
        if not region_mask.any():
            continue
        ys, xs = np.where(region_mask)
        region_score = float(score_map[region_mask].max())
        centroid = (float(xs.mean()), float(ys.mean()))
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        area_pixels = int(region_mask.sum())
        area_percent = 100.0 * area_pixels / object_pixels
        pattern = "broad" if area_percent >= 20.0 else "moderately concentrated" if area_percent >= 7.5 else "compact"
        location = region_location_label(centroid, score_map.shape)
        regions.append(
            {
                "label": location,
                "score": region_score,
                "bbox": bbox,
                "centroid_xy": [round(centroid[0], 1), round(centroid[1], 1)],
                "intensity_label": region_intensity_label(region_score),
                "area_percent": round(area_percent, 3),
                "pattern": pattern,
                "summary": (
                    f"Ranked region in the {location} with {pattern} spread, "
                    f"covering {area_percent:.2f}% of the object and scoring {region_score:.3f}."
                ),
            }
        )

    regions.sort(key=lambda item: item["score"], reverse=True)
    for index, region in enumerate(regions, start=1):
        region["rank"] = index
    return regions[: max(limit, 1)]


def evidence_bullets(package: EvidencePackage) -> List[str]:
    candidates = [
        *package.rgb_observations,
        *package.thermal_observations,
        *package.geometric_observations,
        *package.cross_modal_support,
    ]
    bullets: List[str] = []
    for item in candidates:
        cleaned = str(item).strip().rstrip(".")
        if cleaned and cleaned not in bullets:
            bullets.append(cleaned)
    return bullets[:4]


def resolve_global_retrieval_label(retrieval_status: Mapping[str, Any]) -> str:
    if not retrieval_status.get("configured", False):
        return "RAG unavailable"
    chunk_count = int(retrieval_status.get("chunk_count", 0))
    if chunk_count <= 0:
        return "RAG index empty"
    return f"RAG ready · {chunk_count} chunks"


def resolve_global_gemini_label(
    *,
    gemini_enabled: bool,
    explanation_bundle: ExplanationBundle | None,
) -> str:
    if not gemini_enabled:
        return "Gemini unavailable"
    if explanation_bundle is None:
        return "Gemini configured"
    if explanation_bundle.used_fallback:
        return "Gemini degraded"
    return "Gemini live"


def resolve_explanation_mode_label(
    *,
    retrieval: RetrievalStatus,
    generation: Mapping[str, Any],
) -> str:
    if retrieval.context_count <= 0:
        return "Detector-grounded fallback (RAG-LLM temporarily unavailable)"
    if bool(generation.get("used_fallback", False)):
        return "Grounded by detector + trusted references"
    return "Grounded by detector + trusted references + Gemini"


def build_explanation_bundle(
    *,
    generation: Mapping[str, Any],
    retrieval: RetrievalStatus,
) -> ExplanationBundle:
    return ExplanationBundle(
        mode_label=resolve_explanation_mode_label(retrieval=retrieval, generation=generation),
        explanation_state=str(generation.get("explanation_state", "")),
        report=dict(generation.get("structured_output", {})),
        error_message=str(generation.get("provider_error", "")),
        provider_available=bool(generation.get("provider_available", False)),
        provider_name=str(generation.get("provider", "")),
        model_name=str(generation.get("model", "")),
        used_fallback=bool(generation.get("used_fallback", False)),
        retrieval=retrieval,
        context_pack=dict(generation.get("context_pack", {})),
    )


def build_download_report(
    *,
    package: EvidencePackage,
    regions: Sequence[Mapping[str, Any]],
    explanation: ExplanationBundle,
    source_run_name: str,
    selected_record: ProcessedSampleRecord,
) -> str:
    report = explanation.report
    lines = [
        "# MulSenDiff-X Inspection Report",
        "",
        f"- Category: {package.category}",
        f"- Sample: {selected_record.sample_name}",
        f"- Source model run: {source_run_name}",
        f"- Explanation mode: {explanation.mode_label}",
        f"- Status: {package.status}",
        f"- Anomaly score: {package.raw_score:.6f}",
        f"- Severity: {package.severity_0_100:.1f}/100",
        f"- Affected area: {package.affected_area_pct:.2f}%",
        "",
        "## Detected Regions",
    ]
    if regions:
        for region in regions:
            lines.append(
                f"- {region['label'].title()} — {region['intensity_label']} "
                f"(score {region['score']:.3f}, bbox {region['bbox']})"
            )
    else:
        lines.append("- No localized regions were extracted from the predicted mask.")

    lines.extend(
        [
            "",
            "## Evidence Summary",
            *[f"- {bullet}" for bullet in evidence_bullets(package)],
            "",
            "## Root Cause Explanation",
            f"- Likely cause: {report.get('likely_cause', '')}",
            f"- Why flagged: {report.get('why_flagged', '')}",
            f"- Recommended action: {report.get('recommended_action', '')}",
            f"- Operator summary: {report.get('operator_summary', '')}",
        ]
    )

    citations = report.get("supporting_citations", [])
    if citations:
        lines.append("- Supporting citations:")
        for citation in citations:
            lines.append(
                f"  - {citation.get('title', '')} ({citation.get('source', '')}): "
                f"{citation.get('snippet', '')} [{citation.get('relevance_reason', '')}]"
            )
    else:
        lines.append("- Supporting citations: none")

    if explanation.error_message:
        lines.extend(["", "## Explainability Note", f"- {explanation.error_message}"])
    return "\n".join(lines)


def inspection_state_badge_html(package: EvidencePackage) -> str:
    css_class = "badge-normal" if package.status.lower().startswith("normal") else "badge-anomaly"
    return f'<span class="badge {css_class}">{html.escape(package.status)}</span>'


def explanation_state_badge_html(explanation: ExplanationBundle) -> str:
    state = explanation.explanation_state.lower()
    css_class = "badge-grounded" if state == "grounded" else "badge-cautious"
    label = explanation.mode_label
    if state == "abstained":
        label = f"{label} · cautious"
    elif state == "cautious":
        label = f"{label} · cautious"
    return f'<span class="badge {css_class}">{html.escape(label)}</span>'


def render_top_bar(
    *,
    retrieval_label: str,
    gemini_label: str,
) -> None:
    st.title("MulSenDiff-X Inspector")
    st.caption("Single-page paired-sample inspection with detector-grounded retrieval and Gemini explanation.")
    badge_cols = st.columns(3, gap="small")
    badge_cols[0].info("Model: CCDD demo")
    badge_cols[1].info(retrieval_label)
    badge_cols[2].info(gemini_label)


def render_metadata_strip(record: ProcessedSampleRecord) -> None:
    cols = st.columns(4, gap="small")
    cols[0].metric("Category", record.category)
    cols[1].metric("Defect", record.defect_label)
    cols[2].metric("Split", record.split)
    cols[3].metric("Sample ID", record.sample_id)


def render_metrics_strip(package: EvidencePackage) -> None:
    cols = st.columns(3, gap="small")
    cols[0].metric("Score", f"{package.raw_score:.4f}")
    cols[1].metric("Severity", f"{package.severity_0_100:.1f}/100")
    cols[2].metric("Area", f"{package.affected_area_pct:.2f}%")


def render_visuals(record: ProcessedSampleRecord, visuals: InspectionVisuals) -> None:
    st.subheader("Inspection View")
    image_cols = st.columns(2, gap="medium")
    with image_cols[0]:
        st.image(visuals.original_image, caption="Inspection RGB", use_container_width=False)
    with image_cols[1]:
        st.image(visuals.overlay_image, caption="Localized anomaly overlay", use_container_width=False)
    ref_cols = st.columns([0.28, 0.72], gap="medium")
    with ref_cols[0]:
        if visuals.reference_image is not None:
            st.image(visuals.reference_image, caption="Normal reference", use_container_width=False)
    with ref_cols[1]:
        render_metadata_strip(record)
    st.divider()


def render_decision_card(package: EvidencePackage, explanation: ExplanationBundle) -> None:
    st.subheader("Decision")
    st.write(f"**Status:** {package.status}")
    st.write(f"**Explainability mode:** {explanation.mode_label}")
    render_metrics_strip(package)
    st.divider()


def _render_report_block(title: str, text: str) -> str:
    return (
        '<div class="report-block">'
        f"<h4>{html.escape(title)}</h4>"
        f"<p>{html.escape(text)}</p>"
        "</div>"
    )


def render_explanation_card(explanation: ExplanationBundle) -> None:
    report = explanation.report
    state_message = ""
    if explanation.error_message:
        state_message = explanation.error_message
    elif explanation.retrieval.context_count <= 0:
        state_message = "Trusted references were not retrieved, so this explanation is detector-grounded fallback."
    elif explanation.used_fallback:
        state_message = "Trusted references were retrieved, but Gemini was unavailable, so the report uses grounded fallback text."

    st.subheader("Root-Cause Explanation")
    if state_message:
        st.info(state_message)
    st.markdown(f"**Likely Cause**  \n{report.get('likely_cause', '')}")
    st.markdown(f"**Why Flagged**  \n{report.get('why_flagged', '')}")
    st.markdown(f"**Recommended Action**  \n{report.get('recommended_action', '')}")
    summary = str(report.get("operator_summary", "")).strip()
    if summary:
        st.success(summary)
    st.divider()


def render_evidence_card(evidence: EvidenceSummary) -> None:
    package = evidence.package
    st.subheader("Evidence Summary")
    primary_region = evidence.regions[0] if evidence.regions else None
    if primary_region is not None:
        st.write(
            f"Dominant region: **{primary_region['label']}**, "
            f"{primary_region['intensity_label']}, {float(primary_region['area_percent']):.2f}% of the object."
        )
    else:
        st.write("No connected localization region was extracted from the current mask.")
    st.markdown("**Detector evidence**")
    for item in evidence.bullets:
        st.write(f"- {item}")
    st.caption(
        f"Global descriptor contribution: {package.evidence_breakdown.get('global_descriptor_pct', 0.0):.1f}% · "
        f"score-basis contribution: {package.evidence_breakdown.get('score_basis_pct', 0.0):.1f}%"
    )
    st.divider()


def render_support_card(explanation: ExplanationBundle) -> None:
    citations = explanation.report.get("supporting_citations", [])
    retrieval_items = explanation.retrieval.items
    st.subheader("Supporting References")
    if citations:
        for citation in citations:
            st.markdown(f"**{citation.get('title', '')}**")
            st.caption(f"{citation.get('source', '')} · {citation.get('relevance_reason', '')}")
            st.write(str(citation.get("snippet", "")))
    elif retrieval_items:
        for item in retrieval_items[:3]:
            st.markdown(f"**{item.get('title', '')}**")
            st.caption(f"score {float(item.get('score', 0.0)):.3f} · {item.get('relevance_reason', '')}")
            st.write(str(item.get("snippet", "")))
    else:
        st.info("No trusted reference context was retrieved for this inspection.")
    st.divider()


def render_debug_details(bundle: InspectionBundle) -> None:
    with st.expander("Debug details"):
        st.json(bundle.debug_payload, expanded=False)


def build_inspection_bundle(
    *,
    session: SharedInferenceSession,
    selected_record: ProcessedSampleRecord,
    source_run_name: str,
    known_records: Sequence[ProcessedSampleRecord],
    retrieval_status: Mapping[str, Any],
    gemini_config: Any,
) -> InspectionBundle:
    result = run_known_sample_inference(
        session,
        selected_record,
        output_root=default_app_sessions_root(REPO_ROOT),
        knowledge_base_root=REPO_ROOT / "data" / "retrieval",
    )
    regions = extract_detected_regions(
        predicted_mask=result["predicted_mask"],
        anomaly_map=result["normalized_map"],
        object_mask=result["object_mask"],
    )
    bullets = evidence_bullets(result["package"])
    evidence_payload = build_root_cause_evidence(
        package=result["package"],
        regions=regions,
        evidence_bullets=bullets,
        retrieved_context=result["retrieved_context"],
    )
    generation = generate_root_cause_explanation(
        result["package"],
        config=gemini_config,
        retrieved_context=result["retrieved_context"],
        regions=regions,
        evidence_bullets=bullets,
    )
    generation_paths = save_gemini_generation(
        output_root=result["session_dir"] / "gemini",
        sample_slug=f"{result['package'].category}_{result['package'].defect_label}_{result['package'].sample_id}",
        source_eval_run=session.eval_run_root,
        source_evidence_path=result["package_path"],
        package=result["package"],
        retrieved_context=result["retrieved_context"],
        prompt_payload=generation.get("context_pack", evidence_payload),
        generation=generation,
    )
    retrieval_payload = retrieval_items_to_dicts(list(result.get("retrieved_context", [])))
    retrieval = RetrievalStatus(
        configured=bool(retrieval_status.get("configured", False)),
        chunk_count=int(retrieval_status.get("chunk_count", 0)),
        label=resolve_global_retrieval_label(retrieval_status),
        context_count=len(retrieval_payload),
        items=retrieval_payload,
    )
    explanation = build_explanation_bundle(generation=generation, retrieval=retrieval)
    rgb_array = tensor_to_rgb_array(result["display_rgb"])
    overlay_array = blend_overlay(result["display_rgb"], result["normalized_map"])
    reference_image = load_reference_image(find_reference_record(known_records, result["package"].category))
    visuals = InspectionVisuals(
        original_image=upscale_for_display(rgb_array, target_size=DISPLAY_PANEL_SIZE),
        overlay_image=upscale_for_display(overlay_array, target_size=DISPLAY_PANEL_SIZE),
        reference_image=(
            upscale_for_display(reference_image, target_size=REFERENCE_PANEL_SIZE, sharpen=False)
            if reference_image is not None
            else None
        ),
        panel_path=str(result["panel_path"]),
    )
    evidence = EvidenceSummary(
        package=result["package"],
        regions=regions,
        bullets=bullets,
    )
    report_markdown = build_download_report(
        package=result["package"],
        regions=regions,
        explanation=explanation,
        source_run_name=source_run_name,
        selected_record=selected_record,
    )
    return InspectionBundle(
        selected_record_name=selected_record.sample_name,
        visuals=visuals,
        evidence=evidence,
        explanation=explanation,
        debug_payload={
            "selected_record": selected_record.sample_name,
            "retrieval_query": result.get("retrieval_query", {}),
            "retrieved_context": retrieval_payload,
            "package": result["package"].to_dict(),
            "generation": generation,
            "generation_paths": generation_paths,
        },
        report_markdown=report_markdown,
        session_paths={
            "session_dir": str(result["session_dir"]),
            "panel_path": str(result["panel_path"]),
            "package_path": str(result["package_path"]),
            "retrieved_context_path": str(result["retrieved_context_path"]),
            "retrieval_query_path": str(result["retrieval_query_path"]),
            "json_path": str(generation_paths.get("json_path", "")),
            "markdown_path": str(generation_paths.get("markdown_path", "")),
        },
        metadata={
            "source_run_name": source_run_name,
            "sample_name": selected_record.sample_name,
        },
    )


def main() -> None:
    inject_styles()

    evaluation_runs = find_available_evaluation_runs(REPO_ROOT)
    if not evaluation_runs:
        st.error("No app-ready evaluation runs were found under the repository.")
        return

    selected_run = resolve_primary_demo_run(evaluation_runs)
    bundle = load_eval_bundle_cached(str(selected_run.root))
    gemini_config = load_gemini_config()
    retrieval_root = REPO_ROOT / "data" / "retrieval"
    retrieval_status = load_retrieval_status_cached(str(retrieval_root))

    current_bundle = st.session_state.get("inspection_bundle")
    current_explanation = current_bundle.explanation if isinstance(current_bundle, InspectionBundle) else None

    render_top_bar(
        retrieval_label=resolve_global_retrieval_label(retrieval_status),
        gemini_label=resolve_global_gemini_label(
            gemini_enabled=gemini_config.enabled,
            explanation_bundle=current_explanation,
        ),
    )

    with st.spinner("Loading the curated CCDD demo model and paired sample registry..."):
        session = load_inference_session_cached(str(selected_run.root))
        known_records = load_known_records_cached(
            tuple(session.category_vocabulary),
            str(session.config.get("processed_root") or session.eval_summary.get("processed_root", "")),
        )
        curated_records = build_curated_demo_records(
            preview_selection=bundle.get("preview_selection", {}),
            known_records=known_records,
        )
    if not curated_records:
        st.error("No curated paired samples could be resolved for the CCDD demo.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    categories = categories_in_demo(curated_records)
    if not categories:
        st.error("The curated demo sample set is empty.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "demo_selected_category" not in st.session_state:
        st.session_state["demo_selected_category"] = choose_default_category(categories)

    selected_category = st.session_state["demo_selected_category"]
    category_records = records_for_category(curated_records, selected_category)
    sample_lookup = {record.sample_name: record for record in category_records}
    default_sample_name = category_records[0].sample_name if category_records else ""
    if str(st.session_state.get("demo_selected_sample_name", "")) not in sample_lookup:
        st.session_state["demo_selected_sample_name"] = default_sample_name

    control_cols = st.columns([0.95, 1.05, 0.42, 0.52, 1.06], gap="medium")
    with control_cols[0]:
        selected_category = st.selectbox(
            "Category",
            list(categories),
            key="demo_selected_category",
        )
    category_records = records_for_category(curated_records, selected_category)
    sample_lookup = {record.sample_name: record for record in category_records}
    default_sample_name = category_records[0].sample_name if category_records else ""
    if str(st.session_state.get("demo_selected_sample_name", "")) not in sample_lookup:
        st.session_state["demo_selected_sample_name"] = default_sample_name
    with control_cols[1]:
        selected_sample_name = st.selectbox(
            "Sample",
            list(sample_lookup.keys()),
            format_func=lambda name: format_sample_option(sample_lookup[name]),
            key="demo_selected_sample_name",
        )
    with control_cols[2]:
        inspect_clicked = st.button("Inspect", type="primary", use_container_width=True)
    st.session_state["demo_inspect_clicked"] = inspect_clicked
    selected_record = sample_lookup[selected_sample_name]
    with control_cols[3]:
        if current_bundle is not None:
            st.download_button(
                "Download report",
                data=current_bundle.report_markdown,
                file_name=f"{current_bundle.metadata['sample_name']}_inspection_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
    with control_cols[4]:
        st.markdown("**Demo Contract**")
        st.caption(
            "Curated paired-sample demo. The interface runs the shared CCDD detector and keeps the "
            "full detector -> trusted retrieval -> Gemini chain grounded in RGB, IR, and 3D evidence."
        )

    should_run = bool(st.session_state.get("demo_inspect_clicked", False))

    if should_run:
        with st.spinner("Running paired multi-sensor inspection and grounded explanation..."):
            st.session_state["inspection_bundle"] = build_inspection_bundle(
                session=session,
                selected_record=selected_record,
                source_run_name=selected_run.name,
                known_records=known_records,
                retrieval_status=retrieval_status,
                gemini_config=gemini_config,
            )
        st.session_state["demo_inspect_clicked"] = False
        st.rerun()

    inspection_bundle = st.session_state.get("inspection_bundle")
    if not isinstance(inspection_bundle, InspectionBundle):
        st.info("Choose a curated sample and click `Inspect` to run the paired multi-sensor demo.")
        return

    layout_cols = st.columns([1.16, 0.84], gap="large")
    with layout_cols[0]:
        render_visuals(selected_record, inspection_bundle.visuals)
        render_evidence_card(inspection_bundle.evidence)
    with layout_cols[1]:
        render_decision_card(inspection_bundle.evidence.package, inspection_bundle.explanation)
        render_explanation_card(inspection_bundle.explanation)
        render_support_card(inspection_bundle.explanation)

    render_debug_details(inspection_bundle)


if __name__ == "__main__":
    main()
