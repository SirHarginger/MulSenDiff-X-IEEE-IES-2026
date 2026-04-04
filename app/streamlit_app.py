from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import streamlit as st
import torch
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.dashboard_data import (
    default_app_sessions_root,
    find_available_evaluation_runs,
    find_known_processed_samples,
    load_run_bundle,
    match_uploaded_rgb_name,
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


st.set_page_config(
    page_title="MulSenDiff-X Inspector",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #f7f8fa;
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1220px;
            }
            .app-shell {
                background: transparent;
            }
            .hero h1 {
                margin: 0;
                color: #111827;
                font-size: 2.1rem;
                line-height: 1.1;
                letter-spacing: -0.02em;
            }
            .hero p {
                margin: 0.35rem 0 1.2rem 0;
                color: #4b5563;
                font-size: 1rem;
            }
            .card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
                margin-bottom: 1rem;
            }
            .card h3, .card h4 {
                margin-top: 0;
                color: #111827;
            }
            .status-pill {
                display: inline-block;
                padding: 0.28rem 0.72rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 700;
                margin-bottom: 0.85rem;
                border: 1px solid transparent;
            }
            .status-normal {
                background: #ecfdf3;
                color: #027a48;
                border-color: #abefc6;
            }
            .status-anomaly {
                background: #fff1f3;
                color: #c4320a;
                border-color: #fecdca;
            }
            .bullet-list {
                margin: 0;
                padding-left: 1.1rem;
                color: #374151;
            }
            .subtle {
                color: #6b7280;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_eval_bundle_cached(run_root_str: str) -> Dict[str, Any]:
    return load_run_bundle(run_root_str)


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


def resolve_uploaded_record(
    *,
    upload_name: str,
    upload_bytes: bytes,
    records: Sequence[ProcessedSampleRecord],
) -> ProcessedSampleRecord | None:
    upload_hash = hashlib.sha256(upload_bytes).hexdigest()
    exact_matches: List[ProcessedSampleRecord] = []
    for record in records:
        source_path = Path(record.rgb_source_path)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        if not source_path.exists():
            continue
        if hashlib.sha256(source_path.read_bytes()).hexdigest() == upload_hash:
            exact_matches.append(record)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if exact_matches:
        return exact_matches[0]

    name_matches = match_uploaded_rgb_name(upload_name, records)
    if len(name_matches) == 1:
        return name_matches[0]
    return name_matches[0] if name_matches else None


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
    candidates = [
        Path(record.sample_dir) / "rgb.png",
        Path(record.rgb_source_path),
    ]
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
    return bullets[:3]


def build_download_report(
    *,
    package: EvidencePackage,
    regions: Sequence[Mapping[str, Any]],
    explanation: Mapping[str, Any] | None,
    source_run_name: str,
    selected_record: ProcessedSampleRecord,
) -> str:
    lines = [
        "# MulSenDiff-X Inspection Report",
        "",
        f"- Category: {package.category}",
        f"- Sample: {selected_record.sample_name}",
        f"- Source model run: {source_run_name}",
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
        ]
    )
    if explanation:
        lines.extend(
            [
                f"- Likely cause: {explanation.get('likely_cause', '')}",
                f"- Why flagged: {explanation.get('why_flagged', '')}",
                f"- Recommended action: {explanation.get('recommended_action', '')}",
                f"- Operator summary: {explanation.get('operator_summary', '')}",
            ]
        )
        citations = explanation.get("supporting_citations", [])
        if citations:
            lines.append("- Supporting citations:")
            for citation in citations:
                lines.append(
                    f"  - {citation.get('title', '')} ({citation.get('source', '')}): "
                    f"{citation.get('snippet', '')} [{citation.get('relevance_reason', '')}]"
                )
    else:
        lines.append("- Gemini explanation unavailable for this inspection.")
    return "\n".join(lines)


def render_status_pill(status: str) -> None:
    is_normal = status.lower().startswith("normal")
    css_class = "status-normal" if is_normal else "status-anomaly"
    st.markdown(
        f'<span class="status-pill {css_class}">{status}</span>',
        unsafe_allow_html=True,
    )


def render_result_summary(package: EvidencePackage) -> None:
    cols = st.columns(4)
    cols[0].metric("Status", package.status)
    cols[1].metric("Score", f"{package.raw_score:.4f}")
    cols[2].metric("Severity", f"{package.severity_0_100:.1f}/100")
    cols[3].metric("Affected Area", f"{package.affected_area_pct:.2f}%")


def render_detected_regions(regions: Sequence[Mapping[str, Any]], package: EvidencePackage) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detection Summary")
    if regions:
        primary = regions[0]
        st.write(
            f"The dominant anomaly is concentrated in the **{primary['label']}** "
            f"with {primary['intensity_label']} confidence and affects about {float(primary.get('area_percent', 0.0)):.2f}% of the object."
        )
    elif package.top_regions:
        top_region = package.top_regions[0]
        st.write(
            "The detector identified a compact anomalous region around "
            f"centroid {top_region.get('centroid_xy', [])}."
        )
    else:
        st.write("No localized region summary is available for this inspection.")

    st.markdown("**Detected regions**")
    if regions:
        for region in regions:
            st.write(
                f"- {region['label'].title()} — {region['intensity_label']} "
                f"(score {region['score']:.3f}, area {float(region.get('area_percent', 0.0)):.2f}%)"
            )
    else:
        st.write("- No connected regions extracted from the predicted mask.")

    st.markdown("**Evidence bullets**")
    for bullet in evidence_bullets(package):
        st.write(f"- {bullet}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_root_cause_explanation(
    explanation: Mapping[str, Any] | None,
    error_message: str | None,
    explanation_state: str | None,
) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Root-Cause Explanation")
    if explanation:
        if explanation_state == "abstained":
            st.info("Likely cause is uncertain for this sample. The explanation stays grounded to what the detector can support.")
        elif explanation_state == "cautious":
            st.info("This explanation is cautious because the detector evidence is mixed or only moderately supported.")
        st.markdown(f"**Likely cause**  \n{explanation.get('likely_cause', '')}")
        st.markdown(f"**Why flagged**  \n{explanation.get('why_flagged', '')}")
        st.markdown(f"**Recommended action**  \n{explanation.get('recommended_action', '')}")
        st.caption(str(explanation.get("operator_summary", "")))
        citations = explanation.get("supporting_citations", [])
        if citations:
            st.markdown("**Supporting citations**")
            for citation in citations:
                st.markdown(
                    f"- {citation.get('title', '')} ({citation.get('source', '')}): "
                    f"{citation.get('snippet', '')} [{citation.get('relevance_reason', '')}]"
                )
    else:
        st.warning(error_message or "Gemini explanation is unavailable for this inspection.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_compare_with_normal(reference_image: Image.Image | None, inspection_rgb: np.ndarray) -> None:
    if reference_image is None:
        st.info("No train/good reference image was found for this category.")
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Compare With Normal")
    compare_cols = st.columns(2)
    compare_cols[0].image(inspection_rgb, caption="Inspection image", use_container_width=True)
    compare_cols[1].image(reference_image, caption="Normal reference", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_technical_details(
    *,
    result: Dict[str, Any],
    bundle: Dict[str, Any],
    selected_record: ProcessedSampleRecord,
    regions: Sequence[Mapping[str, Any]],
) -> None:
    package: EvidencePackage = result["package"]
    with st.expander("Technical details"):
        st.markdown("**Model source**")
        st.json(
            {
                "eval_run": str(bundle["run_root"].name),
                "checkpoint_path": bundle["summary"].get("checkpoint_path", ""),
                "masi_calibration_path": bundle["summary"].get("calibration_path", ""),
                "global_descriptor_calibration_path": bundle["summary"].get("global_descriptor_calibration_path", ""),
                "localization_calibration_path": bundle["summary"].get("localization_calibration_path", ""),
                "selected_categories": bundle["summary"].get("selected_categories", []),
                "matched_sample": selected_record.sample_name,
                "session_dir": str(result["session_dir"]),
                "localization_method": str(result.get("localization_method", "")),
            },
            expanded=False,
        )
        st.markdown("**Explainer context pack**")
        st.json(result.get("explanation_context_pack", result["evidence_payload"]), expanded=False)
        st.markdown("**Detected regions**")
        st.json(list(regions), expanded=False)
        st.markdown("**Evidence package**")
        st.json(package.to_dict(), expanded=False)
        if result.get("retrieved_context"):
            st.markdown("**Retrieved context**")
            st.json(
                [item.to_dict() if hasattr(item, "to_dict") else item for item in result["retrieved_context"]],
                expanded=False,
            )
        if result.get("generation_paths"):
            st.markdown("**Saved Gemini outputs**")
            st.json(result["generation_paths"], expanded=False)
        st.markdown("**Explanation support**")
        st.json(
            {
                "explanation_state": result.get("explanation_state", ""),
                "evidence_support_score": result["evidence_payload"].get("confidence", {}).get("evidence_support_score"),
                "uncertainty_signals": result["evidence_payload"].get("uncertainty_signals", []),
            },
            expanded=False,
        )
        st.markdown("**Full technical panel**")
        st.image(str(result["panel_path"]), caption="Full detector panel", use_container_width=True)


def resolve_gemini_status(
    *,
    gemini_enabled: bool,
    inspection_state: Mapping[str, Any] | None,
) -> str:
    if not gemini_enabled:
        return "Gemini: not configured"
    if not inspection_state:
        return "Gemini: key detected"
    if inspection_state.get("generation_paths"):
        return "Gemini: last call succeeded"
    if str(inspection_state.get("explanation_error", "")).strip():
        return "Gemini: key detected, last call failed"
    return "Gemini: key detected"


def main() -> None:
    inject_styles()

    evaluation_runs = find_available_evaluation_runs(REPO_ROOT)
    if not evaluation_runs:
        st.error("No app-ready model bundles or evaluation runs were found under model/ or runs/.")
        return

    st.sidebar.subheader("Model Source")
    run_lookup = {run.name: run for run in evaluation_runs}
    selected_run_name = st.sidebar.selectbox("Evaluation run", list(run_lookup.keys()))
    selected_run = run_lookup[selected_run_name]
    bundle = load_eval_bundle_cached(str(selected_run.root))
    session = load_inference_session_cached(str(selected_run.root))
    known_records = load_known_records_cached(
        tuple(session.category_vocabulary),
        str(session.config.get("processed_root") or session.eval_summary.get("processed_root", "")),
    )
    gemini_config = load_gemini_config()
    inspection_state = st.session_state.get("inspection_result")
    st.sidebar.caption(
        resolve_gemini_status(
            gemini_enabled=gemini_config.enabled,
            inspection_state=inspection_state if isinstance(inspection_state, Mapping) else None,
        )
    )
    st.sidebar.caption(
        "This demo resolves uploads against known paired samples so the detector can use its full multimodal backend."
    )

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("## MulSenDiff-X Inspector")
    st.markdown("Upload an image. Detect anomalies. Explain the likely root cause.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_cols = st.columns([1.8, 1.0, 0.8], vertical_alignment="bottom")
    uploaded_file = input_cols[0].file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    selected_category = input_cols[1].selectbox("Category", list(session.category_vocabulary))
    run_clicked = input_cols[2].button("Run Inspection", type="primary", use_container_width=True)
    st.caption("Paired-sample demo: uploads must match a known RGB sample so the backend can resolve the paired modalities.")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
        if uploaded_file is None:
            st.error("Upload a category-matching RGB image to run the inspection.")
        else:
            candidate_records = [record for record in known_records if record.category == selected_category]
            selected_record = resolve_uploaded_record(
                upload_name=uploaded_file.name,
                upload_bytes=uploaded_file.getvalue(),
                records=candidate_records,
            )
            if selected_record is None:
                st.error(
                    "The uploaded image could not be matched to a known paired sample for this category. "
                    "Use a dataset RGB file so the demo can resolve the paired modalities."
                )
            else:
                with st.spinner("Running anomaly inspection..."):
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
                    evidence_payload = build_root_cause_evidence(
                        package=result["package"],
                        regions=regions,
                        evidence_bullets=evidence_bullets(result["package"]),
                        retrieved_context=result["retrieved_context"],
                    )

                    explanation_output: Dict[str, Any] | None = None
                    explanation_error = ""
                    generation_paths: Dict[str, str] | None = None
                    explanation_context_pack: Dict[str, Any] | None = None
                    explanation_state = evidence_payload.get("confidence", {}).get("recommended_explanation_state", "")
                    try:
                        generation = generate_root_cause_explanation(
                            result["package"],
                            config=gemini_config,
                            retrieved_context=result["retrieved_context"],
                            regions=regions,
                            evidence_bullets=evidence_bullets(result["package"]),
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
                        explanation_output = dict(generation["structured_output"])
                        explanation_context_pack = dict(generation.get("context_pack", {}))
                        explanation_state = str(generation.get("explanation_state", explanation_state))
                    except Exception as exc:
                        explanation_error = str(exc)

                    st.session_state["inspection_result"] = {
                        "selected_record": selected_record,
                        "result": result,
                        "regions": regions,
                        "evidence_payload": evidence_payload,
                        "explanation_context_pack": explanation_context_pack,
                        "explanation_output": explanation_output,
                        "explanation_state": explanation_state,
                        "explanation_error": explanation_error,
                        "generation_paths": generation_paths,
                    }

    inspection_state = st.session_state.get("inspection_result")
    if not inspection_state:
        st.info("Run an inspection to see the anomaly overlay, severity summary, and Gemini explanation.")
        return

    selected_record = inspection_state["selected_record"]
    result = inspection_state["result"]
    package: EvidencePackage = result["package"]
    regions = inspection_state["regions"]
    explanation_output = inspection_state["explanation_output"]
    explanation_state = inspection_state.get("explanation_state", "")
    explanation_error = inspection_state["explanation_error"]
    generation_paths = inspection_state["generation_paths"]
    original_image = tensor_to_rgb_array(result["display_rgb"])
    overlay_image = blend_overlay(result["display_rgb"], result["normalized_map"])
    reference_image = load_reference_image(find_reference_record(known_records, package.category))

    render_status_pill(package.status)
    render_result_summary(package)

    evidence_cols = st.columns(2, gap="large")
    with evidence_cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Original Image")
        st.image(original_image, caption="Inspection image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with evidence_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Anomaly Map / Overlay")
        st.image(overlay_image, caption="Localized anomaly overlay", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    render_detected_regions(regions, package)
    render_root_cause_explanation(explanation_output, explanation_error, explanation_state)

    actions = st.columns([1.0, 1.0])
    compare_with_normal = actions[0].toggle("Compare with normal")
    report_text = build_download_report(
        package=package,
        regions=regions,
        explanation=explanation_output,
        source_run_name=selected_run.name,
        selected_record=selected_record,
    )
    actions[1].download_button(
        "Download report",
        data=report_text,
        file_name=f"{package.category}_{package.sample_id}_inspection_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if compare_with_normal:
        render_compare_with_normal(reference_image, original_image)

    render_technical_details(
        result={
            **result,
            "evidence_payload": inspection_state["evidence_payload"],
            "generation_paths": generation_paths,
        },
        bundle=bundle,
        selected_record=selected_record,
        regions=regions,
    )


if __name__ == "__main__":
    main()
