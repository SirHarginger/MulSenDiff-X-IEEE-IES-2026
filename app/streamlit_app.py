from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DescriptorConditioningDataset, collate_model_samples
from src.evaluation.metrics import binary_f1_score, intersection_over_union
from src.explainer.evidence_builder import build_evidence_package
from src.explainer.report import render_templated_explanation
from src.inference.anomaly_scorer import score_batch
from src.inference.localization import estimate_object_mask, normalize_anomaly_map, threshold_anomaly_map
from src.inference.quantification import fit_masi_calibration
from app.dashboard_data import (
    find_available_training_runs,
    list_checkpoint_files,
    load_json,
)
from src.models.mulsendiffx import MulSenDiffX
from src.training.train_diffusion import collect_reference_scores, load_gt_mask
from src.utils.checkpoint import load_checkpoint


st.set_page_config(
    page_title="MulSenDiff-X Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(196, 109, 28, 0.12), transparent 26%),
                    linear-gradient(180deg, #f4f1ea 0%, #fbfaf7 100%);
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }
            .hero-card {
                background: linear-gradient(135deg, #1d2a33 0%, #314754 100%);
                color: #f6f1e7;
                padding: 1.2rem 1.3rem;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 18px 48px rgba(24, 33, 40, 0.12);
                margin-bottom: 1rem;
            }
            .hero-card h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: 0.02em;
            }
            .hero-card p {
                margin: 0.4rem 0 0 0;
                color: #d6d9dc;
            }
            .section-card {
                background: rgba(255,255,255,0.78);
                border: 1px solid rgba(40,52,60,0.08);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 28px rgba(20, 28, 36, 0.05);
            }
            .pill {
                display: inline-block;
                padding: 0.2rem 0.65rem;
                border-radius: 999px;
                background: #d66c1f;
                color: white;
                font-size: 0.86rem;
                font-weight: 600;
                margin-right: 0.4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_inference_session(run_root_str: str, checkpoint_str: str) -> Dict[str, Any]:
    run_root = Path(run_root_str)
    checkpoint_path = Path(checkpoint_str)
    config = load_json(run_root / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config payload for run {run_root}")

    target_size = tuple(int(value) for value in config.get("target_size", [256, 256]))
    device = torch.device(str(config.get("device", "cpu")))
    batch_size = int(config.get("batch_size", 4))
    score_mode = str(config.get("score_mode", "noise_error"))
    category = str(config.get("category", "capsule"))

    train_dataset = DescriptorConditioningDataset(
        config["train_manifest_csv"],
        data_root=REPO_ROOT,
        target_size=target_size,
        global_stats_path=config["global_stats_json"],
    )
    eval_dataset = DescriptorConditioningDataset(
        config["eval_manifest_csv"],
        data_root=REPO_ROOT,
        target_size=target_size,
        global_stats_path=config["global_stats_json"],
    )

    model = MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=train_dataset.descriptor_channels,
        global_dim=train_dataset.global_dim,
    ).to(device)
    checkpoint_payload = load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    calibration_scores = collect_reference_scores(
        model,
        train_loader,
        device=device,
        score_mode=score_mode,
        anomaly_timestep=int(config.get("anomaly_timestep", 200)),
        descriptor_weight=float(config.get("descriptor_weight", 0.25)),
        object_mask_threshold=float(config.get("object_mask_threshold", 0.02)),
    )
    calibration = fit_masi_calibration(calibration_scores, category=category, score_mode=score_mode)

    return {
        "run_root": run_root,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "device": device,
        "target_size": target_size,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "calibration": calibration,
        "checkpoint_payload": checkpoint_payload,
    }


def run_live_sample_inference(session: Dict[str, Any], sample_index: int) -> Dict[str, Any]:
    sample = session["eval_dataset"][sample_index]
    config = session["config"]
    device = session["device"]
    model = session["model"]

    x_rgb = sample.x_rgb.unsqueeze(0).to(device)
    descriptor_maps = sample.conditioning.descriptor_maps.unsqueeze(0).to(device)
    global_vector = sample.conditioning.global_vector.unsqueeze(0).to(device)

    scored = score_batch(
        model,
        x_rgb,
        descriptor_maps,
        global_vector,
        score_mode=str(config.get("score_mode", "noise_error")),
        timestep=int(config.get("anomaly_timestep", 200)),
        descriptor_weight=float(config.get("descriptor_weight", 0.25)),
        object_mask_threshold=float(config.get("object_mask_threshold", 0.02)),
    )

    normalized_map = normalize_anomaly_map(scored.anomaly_map.detach().cpu())
    object_mask = estimate_object_mask(sample.x_rgb.unsqueeze(0), threshold=float(config.get("object_mask_threshold", 0.02)))
    predicted_mask = threshold_anomaly_map(
        normalized_map,
        percentile=float(config.get("pixel_threshold_percentile", 0.9)),
        object_mask=object_mask,
        normalized=True,
    )
    gt_mask = load_gt_mask(sample.rgb_gt_path, target_size=session["target_size"])
    package = build_evidence_package(
        category=sample.category,
        split=sample.split,
        defect_label=sample.defect_label,
        sample_id=sample.sample_id,
        is_anomalous=sample.is_anomalous,
        score_mode=str(config.get("score_mode", "noise_error")),
        score_label=scored.score_basis_label,
        raw_score=float(scored.anomaly_score[0].item()),
        calibration=session["calibration"],
        normalized_anomaly_map=normalized_map[0],
        score_basis_map=scored.score_basis_map[0].detach().cpu(),
        descriptor_support=scored.descriptor_support[0].detach().cpu(),
        object_mask=object_mask[0],
        predicted_mask=predicted_mask[0],
        descriptor_maps=sample.conditioning.descriptor_maps.detach().cpu(),
        source_paths={
            "rgb_path": sample.rgb_path,
            "rgb_gt_path": sample.rgb_gt_path,
            "ir_gt_path": sample.ir_gt_path,
            "pointcloud_gt_path": sample.pointcloud_gt_path,
            "visualization_path": "",
        },
    )
    report_text = render_templated_explanation(package)

    pixel_iou = None
    pixel_f1 = None
    if gt_mask is not None:
        pixel_iou = intersection_over_union(predicted_mask[0], gt_mask)
        pixel_f1 = binary_f1_score(predicted_mask[0], gt_mask)

    return {
        "sample": sample,
        "scored": scored,
        "normalized_map": normalized_map[0],
        "object_mask": object_mask[0],
        "predicted_mask": predicted_mask[0],
        "gt_mask": gt_mask,
        "package": package,
        "report_text": report_text,
        "pixel_iou": pixel_iou,
        "pixel_f1": pixel_f1,
    }


def tensor_to_rgb_image(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def tensor_to_gray_image(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().squeeze().numpy()


def main() -> None:
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <h1>MulSenDiff-X Inspector</h1>
            <p>Checkpoint-driven multimodal anomaly review for trained runs, with live inference, calibrated severity, and evidence-grounded reporting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    training_runs = find_available_training_runs(REPO_ROOT / "runs")
    if not training_runs:
        st.error("No trained runs with checkpoints were found under runs/. Train a model first.")
        return

    st.sidebar.header("Run Controls")
    run_lookup = {run.name: run for run in training_runs}
    selected_run_name = st.sidebar.selectbox("Training Run", list(run_lookup.keys()))
    selected_run = run_lookup[selected_run_name]
    checkpoint_files = list_checkpoint_files(selected_run.root)
    checkpoint_lookup = {path.name: path for path in checkpoint_files}
    selected_checkpoint_name = st.sidebar.selectbox(
        "Checkpoint",
        list(checkpoint_lookup.keys()),
        index=list(checkpoint_lookup.keys()).index("best.pt") if "best.pt" in checkpoint_lookup else 0,
    )

    session = load_inference_session(str(selected_run.root), str(checkpoint_lookup[selected_checkpoint_name]))
    eval_dataset = session["eval_dataset"]
    config = session["config"]

    eval_rows = eval_dataset.rows
    defect_options = sorted({row.defect_label for row in eval_rows})
    chosen_defect = st.sidebar.selectbox("Defect Filter", ["All"] + defect_options)
    show_anomalous_only = st.sidebar.toggle("Show Anomalous Only", value=False)

    visible_indices: List[int] = []
    visible_labels: List[str] = []
    for index, row in enumerate(eval_rows):
        if chosen_defect != "All" and row.defect_label != chosen_defect:
            continue
        if show_anomalous_only and not row.is_anomalous:
            continue
        visible_indices.append(index)
        label = f"{row.defect_label} / {row.sample_id} | {'anomalous' if row.is_anomalous else 'good'}"
        visible_labels.append(label)

    if not visible_indices:
        st.warning("No samples match the current filters.")
        return

    selected_position = st.sidebar.selectbox(
        "Sample",
        list(range(len(visible_labels))),
        format_func=lambda idx: visible_labels[idx],
    )
    inference = run_live_sample_inference(session, visible_indices[selected_position])
    package = inference["package"]
    sample = inference["sample"]
    scored = inference["scored"]

    st.markdown(
        f'<span class="pill">{package.status}</span><span class="pill">{package.score_mode}</span><span class="pill">{sample.category}</span>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Severity", f"{package.severity_0_100:.1f}/100")
    metric_cols[1].metric("Confidence", f"{package.confidence_0_100:.1f}/100")
    metric_cols[2].metric("Raw Score", f"{package.raw_score:.4f}")
    metric_cols[3].metric("Affected Area", f"{package.affected_area_pct:.2f}%")
    metric_cols[4].metric("Peak Anomaly", f"{package.peak_anomaly_0_100:.1f}/100")

    left, right = st.columns([1.45, 1.0], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Live Inference View")
        image_cols = st.columns(3)
        image_cols[0].image(tensor_to_rgb_image(sample.x_rgb), caption="RGB", use_container_width=True)
        image_cols[1].image(
            tensor_to_rgb_image(scored.reconstructed_rgb[0].detach().cpu()),
            caption="Reconstructed",
            use_container_width=True,
        )
        image_cols[2].image(tensor_to_gray_image(inference["normalized_map"]), caption="Anomaly Map", use_container_width=True)

        map_cols = st.columns(4)
        map_cols[0].image(tensor_to_gray_image(scored.score_basis_map[0].detach().cpu()), caption=scored.score_basis_label, use_container_width=True)
        map_cols[1].image(tensor_to_gray_image(scored.descriptor_support[0].detach().cpu()), caption="Descriptor Support", use_container_width=True)
        map_cols[2].image(tensor_to_gray_image(inference["predicted_mask"]), caption="Predicted Mask", use_container_width=True)
        if inference["gt_mask"] is not None:
            map_cols[3].image(tensor_to_gray_image(inference["gt_mask"]), caption="GT Mask", use_container_width=True)
        else:
            map_cols[3].info("No GT mask available for this sample.")

        if inference["pixel_iou"] is not None and inference["pixel_f1"] is not None:
            st.caption(f"Live localisation check: Pixel IoU {inference['pixel_iou']:.4f} | Pixel F1 {inference['pixel_f1']:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Templated Grounded Report")
        st.code(inference["report_text"], language="markdown")
        report_bytes = inference["report_text"].encode("utf-8")
        st.download_button("Download Report", data=report_bytes, file_name=f"{sample.category}_{sample.defect_label}_{sample.sample_id}.md", mime="text/markdown")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Evidence Breakdown")
        breakdown_rows = [{"source": key.replace("_pct", "").replace("_", " ").title(), "percent": value} for key, value in package.evidence_breakdown.items()]
        st.bar_chart({row["source"]: row["percent"] for row in breakdown_rows}, horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Observations")
        for section_title, items in [
            ("RGB / Score Basis", package.rgb_observations),
            ("Thermal", package.thermal_observations),
            ("Geometry", package.geometric_observations),
            ("Cross-Modal", package.cross_modal_support),
            ("Confidence Notes", package.confidence_notes),
        ]:
            st.markdown(f"**{section_title}**")
            for item in items:
                st.write(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Run Metadata")
        top_region = package.top_regions[0] if package.top_regions else {}
        st.json(
            {
                "run": selected_run.root.name,
                "checkpoint": selected_checkpoint_name,
                "loaded_epoch": int(session["checkpoint_payload"].get("epoch", 0)),
                "category": config.get("category"),
                "score_mode": config.get("score_mode"),
                "anomaly_timestep": config.get("anomaly_timestep"),
                "calibration": session["calibration"].to_dict(),
                "top_region": top_region,
                "source_paths": package.source_paths,
            },
            expanded=False,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Filtered Sample Inventory")
    inventory_rows = []
    for index in visible_indices:
        row = eval_rows[index]
        inventory_rows.append(
            {
                "defect_label": row.defect_label,
                "sample_id": row.sample_id,
                "is_anomalous": row.is_anomalous,
                "rgb_path": row.rgb_path,
            }
        )
    st.dataframe(inventory_rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
