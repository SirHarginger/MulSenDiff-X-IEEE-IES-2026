from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageStat


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
        issues.append(
            DescriptorValidationIssue(artifact.name, "error", f"unsupported_descriptor_kind={artifact.kind}")
        )
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
    elif artifact.kind == "global":
        if artifact.vector_length <= 0:
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
        elif artifact.kind == "global":
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
    payload = [asdict(artifact) for artifact in artifacts]
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_descriptor_manifest(path: Path | str) -> List[DescriptorArtifact]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DescriptorArtifact(**item) for item in payload]


def _find_artifact(artifacts: Sequence[DescriptorArtifact], name: str) -> DescriptorArtifact:
    for artifact in artifacts:
        if artifact.name == name:
            return artifact
    raise KeyError(name)


def _binary_union(left: Image.Image, right: Image.Image) -> Image.Image:
    return ImageChops.lighter(left.convert("L"), right.convert("L"))


def _binary_overlap_ratio(left: Image.Image, right: Image.Image) -> float:
    left_pixels = list(left.convert("L").getdata())
    right_pixels = list(right.convert("L").getdata())
    intersection = 0
    union = 0
    for l_value, r_value in zip(left_pixels, right_pixels):
        l_active = l_value > 0
        r_active = r_value > 0
        intersection += int(l_active and r_active)
        union += int(l_active or r_active)
    if union == 0:
        return 0.0
    return intersection / union


def _mask_centroid(mask: Image.Image) -> Tuple[float, float] | None:
    mask = mask.convert("L")
    width, height = mask.size
    sum_x = 0.0
    sum_y = 0.0
    count = 0.0
    for index, value in enumerate(mask.getdata()):
        if value > 0:
            sum_x += index % width
            sum_y += index // width
            count += 1.0
    if count == 0.0:
        return None
    return (sum_x / count, sum_y / count)


def _centroid_distance(left: Image.Image, right: Image.Image) -> float:
    left_centroid = _mask_centroid(left)
    right_centroid = _mask_centroid(right)
    if left_centroid is None or right_centroid is None:
        return 1.0
    width, height = left.size
    diagonal = max((width**2 + height**2) ** 0.5, 1.0)
    dx = left_centroid[0] - right_centroid[0]
    dy = left_centroid[1] - right_centroid[1]
    return ((dx**2 + dy**2) ** 0.5) / diagonal


def generate_crossmodal_descriptor_bundle(
    rgb_path: Path | str,
    infrared_manifest_path: Path | str,
    pointcloud_manifest_path: Path | str,
    out_dir: Path | str,
) -> Dict[str, object]:
    rgb_path = Path(rgb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_size = Image.open(rgb_path).size
    ir_artifacts = _load_descriptor_manifest(infrared_manifest_path)
    pc_artifacts = _load_descriptor_manifest(pointcloud_manifest_path)

    ir_hotspot = Image.open(_find_artifact(ir_artifacts, "infrared_hotspot").path).convert("L").resize(rgb_size)
    pc_curvature = Image.open(_find_artifact(pc_artifacts, "pointcloud_curvature").path).convert("L").resize(rgb_size)
    pc_roughness = Image.open(_find_artifact(pc_artifacts, "pointcloud_roughness").path).convert("L").resize(rgb_size)
    pc_normal_deviation = (
        Image.open(_find_artifact(pc_artifacts, "pointcloud_normal_deviation").path).convert("L").resize(rgb_size)
    )
    pc_density = Image.open(_find_artifact(pc_artifacts, "pointcloud_density_topdown").path).convert("L").resize(rgb_size)

    geometric_suspicion = ImageChops.lighter(pc_curvature, ImageChops.lighter(pc_roughness, pc_normal_deviation))
    agreement = ImageChops.multiply(ir_hotspot, geometric_suspicion)
    support = ImageChops.lighter(pc_density, agreement)
    inconsistency = ImageChops.difference(ir_hotspot, geometric_suspicion)

    ir_support_path = out_dir / "infrared_support.png"
    geometric_support_path = out_dir / "geometric_support.png"
    agreement_path = out_dir / "agreement.png"
    inconsistency_path = out_dir / "inconsistency.png"
    global_path = out_dir / "global.json"
    manifest_path = out_dir / "manifest.json"

    ir_hotspot.save(ir_support_path)
    support.save(geometric_support_path)
    agreement.save(agreement_path)
    inconsistency.save(inconsistency_path)

    agreement_stat = ImageStat.Stat(agreement)
    support_stat = ImageStat.Stat(support)
    inconsistency_stat = ImageStat.Stat(inconsistency)
    overlap_score = _binary_overlap_ratio(ir_hotspot, geometric_suspicion)
    centroid_distance = _centroid_distance(ir_hotspot, geometric_suspicion)
    global_summary = {
        "rgb_width": rgb_size[0],
        "rgb_height": rgb_size[1],
        "agreement_mean": round(agreement_stat.mean[0], 6),
        "agreement_std": round(agreement_stat.stddev[0], 6),
        "support_mean": round(support_stat.mean[0], 6),
        "ir_pc_overlap_score": round(overlap_score, 6),
        "ir_pc_centroid_distance": round(centroid_distance, 6),
        "crossmodal_inconsistency_score": round(inconsistency_stat.mean[0] / 255.0, 6),
        "source_rgb_path": str(rgb_path),
    }
    global_path.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")

    artifacts = [
        build_descriptor_metadata(
            name="crossmodal_infrared_support",
            kind="spatial",
            path=ir_support_path,
            channels=1,
            width=rgb_size[0],
            height=rgb_size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="crossmodal_geometric_support",
            kind="spatial",
            path=geometric_support_path,
            channels=1,
            width=rgb_size[0],
            height=rgb_size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="crossmodal_agreement",
            kind="spatial",
            path=agreement_path,
            channels=1,
            width=rgb_size[0],
            height=rgb_size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="crossmodal_inconsistency",
            kind="spatial",
            path=inconsistency_path,
            channels=1,
            width=rgb_size[0],
            height=rgb_size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="crossmodal_global",
            kind="global",
            path=global_path,
            vector_length=6,
            dtype="json",
        ),
    ]
    write_descriptor_manifest(artifacts, manifest_path)
    return {"artifacts": artifacts, "manifest_path": manifest_path, "global_summary": global_summary}
