from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DescriptorArtifact:
    name: str
    kind: str
    path: str
    channels: int = 0
    width: int = 0
    height: int = 0
    vector_length: int = 0
    dtype: str = ""


@dataclass(frozen=True)
class DescriptorValidationIssue:
    artifact: str
    severity: str
    message: str


def build_descriptor_metadata(
    *,
    name: str,
    kind: str,
    path: Path | str,
    channels: int = 0,
    width: int = 0,
    height: int = 0,
    vector_length: int = 0,
    dtype: str = "",
) -> DescriptorArtifact:
    return DescriptorArtifact(
        name=name,
        kind=kind,
        path=str(path),
        channels=channels,
        width=width,
        height=height,
        vector_length=vector_length,
        dtype=dtype,
    )


def validate_descriptor_artifact(artifact: DescriptorArtifact) -> List[DescriptorValidationIssue]:
    issues: List[DescriptorValidationIssue] = []
    path = Path(artifact.path)
    if artifact.kind not in {"spatial", "global"}:
        issues.append(DescriptorValidationIssue(artifact.name, "error", f"unsupported_descriptor_kind={artifact.kind}"))
        return issues
    if not path.exists():
        issues.append(DescriptorValidationIssue(artifact.name, "error", f"missing_file={path}"))
        return issues
    if path.stat().st_size <= 0:
        issues.append(DescriptorValidationIssue(artifact.name, "error", "empty_descriptor_file"))
    if artifact.kind == "spatial":
        if artifact.width <= 0 or artifact.height <= 0:
            issues.append(DescriptorValidationIssue(artifact.name, "error", "invalid_spatial_shape"))
        if artifact.channels <= 0:
            issues.append(DescriptorValidationIssue(artifact.name, "error", "missing_spatial_channels"))
    if artifact.kind == "global" and artifact.vector_length <= 0:
        issues.append(DescriptorValidationIssue(artifact.name, "error", "invalid_global_vector_length"))
    if not artifact.dtype:
        issues.append(DescriptorValidationIssue(artifact.name, "warning", "missing_dtype_metadata"))
    return issues


def validate_descriptor_bundle(
    artifacts: Sequence[DescriptorArtifact],
    *,
    require_spatial: bool = True,
    require_global: bool = True,
) -> List[DescriptorValidationIssue]:
    issues: List[DescriptorValidationIssue] = []
    spatial_count = 0
    global_count = 0
    for artifact in artifacts:
        if artifact.kind == "spatial":
            spatial_count += 1
        if artifact.kind == "global":
            global_count += 1
        issues.extend(validate_descriptor_artifact(artifact))
    if require_spatial and spatial_count == 0:
        issues.append(DescriptorValidationIssue("bundle", "error", "missing_spatial_descriptor"))
    if require_global and global_count == 0:
        issues.append(DescriptorValidationIssue("bundle", "error", "missing_global_descriptor"))
    return issues


def descriptor_bundle_is_valid(artifacts: Sequence[DescriptorArtifact]) -> bool:
    return not any(issue.severity == "error" for issue in validate_descriptor_bundle(artifacts))


def write_descriptor_manifest(artifacts: Sequence[DescriptorArtifact], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(artifact) for artifact in artifacts], indent=2, sort_keys=True), encoding="utf-8")


def resize_to_rgb_grid(array: np.ndarray, *, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    image = image.resize(target_size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def verify_crossmodal_alignment(
    ir_normalized: np.ndarray,
    pc_depth: np.ndarray,
    *,
    tolerance_px: int = 5,
) -> Dict[str, object]:
    ir_mask = ir_normalized > 0.02
    pc_mask = pc_depth > 0.02
    ir_bbox = _bbox(ir_mask) or _full_frame_bbox(ir_normalized.shape)
    pc_bbox = _bbox(pc_mask) or _full_frame_bbox(pc_depth.shape)

    deltas = [abs(a - b) for a, b in zip(ir_bbox, pc_bbox)]
    max_delta = max(deltas) if deltas else 0
    return {
        "passed": max_delta <= tolerance_px,
        "message": "ok" if max_delta <= tolerance_px else "boundary_overlap_exceeds_tolerance",
        "ir_bbox": ir_bbox,
        "pc_bbox": pc_bbox,
        "max_delta": max_delta,
    }


def generate_crossmodal_maps(
    *,
    ir_hotspot: np.ndarray,
    ir_gradient: np.ndarray,
    pc_curvature: np.ndarray,
    pc_roughness: np.ndarray,
    ir_normalized: np.ndarray,
    pc_depth: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    tolerance_px: int = 5,
) -> Dict[str, object]:
    alignment = verify_crossmodal_alignment(ir_normalized, pc_depth, tolerance_px=tolerance_px)

    cross_ir_support = np.maximum(
        resize_to_rgb_grid(ir_hotspot, target_size=target_size),
        resize_to_rgb_grid(ir_gradient, target_size=target_size),
    ).astype(np.float32)
    cross_geo_support = np.maximum(
        resize_to_rgb_grid(pc_curvature, target_size=target_size),
        resize_to_rgb_grid(pc_roughness, target_size=target_size),
    ).astype(np.float32)
    cross_agreement = np.clip(cross_ir_support * cross_geo_support, 0.0, 1.0).astype(np.float32)
    cross_inconsistency = np.abs(cross_ir_support - cross_geo_support).astype(np.float32)

    return {
        "cross_ir_support": cross_ir_support,
        "cross_geo_support": cross_geo_support,
        "cross_agreement": cross_agreement,
        "cross_inconsistency": cross_inconsistency,
        "global": {
            "cross_overlap_score": round(float(cross_agreement.mean()), 6),
            "cross_centroid_distance": round(
                _top_percent_centroid_distance(cross_ir_support, cross_geo_support, percentile=95.0),
                6,
            ),
            "cross_inconsistency_score": round(float(cross_inconsistency.mean()), 6),
            "cross_alignment_passed": bool(alignment["passed"]),
            "cross_alignment_message": str(alignment["message"]),
            "cross_alignment_max_delta": int(alignment["max_delta"]),
        },
        "alignment": alignment,
    }


def save_grayscale_png(array: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(path)


def save_preview_png(
    array: np.ndarray,
    path: Path | str,
    *,
    upper_percentile: float = 99.5,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
    nonzero = clipped[clipped > 0.0]
    if nonzero.size == 0:
        preview = np.zeros_like(clipped, dtype=np.float32)
    else:
        upper = float(np.percentile(nonzero, upper_percentile))
        preview = np.clip(clipped / max(upper, 1e-6), 0.0, 1.0).astype(np.float32)

    Image.fromarray((preview * 255.0).round().astype(np.uint8), mode="L").save(path)


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    ys = coordinates[:, 0]
    xs = coordinates[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _full_frame_bbox(shape: tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = shape
    return 0, 0, max(width - 1, 0), max(height - 1, 0)


def _top_percent_centroid_distance(left: np.ndarray, right: np.ndarray, *, percentile: float) -> float:
    left_centroid = _top_percent_centroid(left, percentile=percentile)
    right_centroid = _top_percent_centroid(right, percentile=percentile)
    if left_centroid is None or right_centroid is None:
        return 1.0
    height, width = left.shape
    diagonal = max(float((height * height + width * width) ** 0.5), 1e-6)
    dx = float(left_centroid[0] - right_centroid[0])
    dy = float(left_centroid[1] - right_centroid[1])
    return float(((dx * dx + dy * dy) ** 0.5) / diagonal)


def _top_percent_centroid(array: np.ndarray, *, percentile: float) -> tuple[float, float] | None:
    if array.size == 0:
        return None
    threshold = float(np.percentile(array, percentile))
    mask = array >= threshold
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    ys = coordinates[:, 0].astype(np.float32)
    xs = coordinates[:, 1].astype(np.float32)
    return float(xs.mean()), float(ys.mean())
