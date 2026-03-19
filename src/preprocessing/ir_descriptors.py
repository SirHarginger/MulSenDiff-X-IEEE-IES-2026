from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageStat

from src.preprocessing.crossmodal_descriptors import build_descriptor_metadata, write_descriptor_manifest


@dataclass(frozen=True)
class InfraredImageSummary:
    path: str
    mode: str
    width: int
    height: int
    mean_intensity: Tuple[float, ...]
    extrema: Tuple[Tuple[int, int], ...]


def summarize_infrared_image(path: Path | str) -> InfraredImageSummary:
    path = Path(path)
    image = Image.open(path)
    stat = ImageStat.Stat(image)
    extrema = image.getextrema()
    if isinstance(extrema[0], int):
        extrema = (extrema,)  # type: ignore[assignment]
    return InfraredImageSummary(
        path=str(path),
        mode=image.mode,
        width=image.size[0],
        height=image.size[1],
        mean_intensity=tuple(round(value, 4) for value in stat.mean),
        extrema=tuple(extrema),  # type: ignore[arg-type]
    )


def _normalize_grayscale(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    minimum, maximum = image.getextrema()
    if maximum <= minimum:
        return Image.new("L", image.size, 0)
    scale = 255.0 / (maximum - minimum)
    return image.point(lambda pixel: int((pixel - minimum) * scale))


def _histogram_percentile(image: Image.Image, percentile: float) -> int:
    image = image.convert("L")
    histogram = image.histogram()
    total = sum(histogram)
    target = total * percentile
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return value
    return 255


def _gradient_magnitude(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    dx = ImageChops.difference(image, ImageChops.offset(image, 1, 0))
    dy = ImageChops.difference(image, ImageChops.offset(image, 0, 1))
    return ImageChops.lighter(dx, dy)


def _local_variance_proxy(image: Image.Image, radius: int = 5) -> Image.Image:
    image = image.convert("L")
    local_mean = image.filter(ImageFilter.BoxBlur(radius))
    local_difference = ImageChops.difference(image, local_mean)
    return local_difference.filter(ImageFilter.BoxBlur(max(1, radius // 2)))


def _masked_mean(image: Image.Image, mask: Image.Image) -> float:
    pixels = list(image.convert("L").getdata())
    mask_pixels = list(mask.convert("L").getdata())
    selected = [value for value, enabled in zip(pixels, mask_pixels) if enabled > 0]
    if not selected:
        return 0.0
    return sum(selected) / len(selected)


def _mask_compactness(mask: Image.Image) -> float:
    width, height = mask.size
    coords: List[Tuple[int, int]] = []
    for index, value in enumerate(mask.convert("L").getdata()):
        if value > 0:
            x = index % width
            y = index // width
            coords.append((x, y))

    if not coords:
        return 0.0

    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    bbox_area = max((max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1), 1)
    return len(coords) / bbox_area


def generate_ir_descriptor_bundle(ir_path: Path | str, out_dir: Path | str) -> Dict[str, object]:
    ir_path = Path(ir_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_image = Image.open(ir_path).convert("L")
    normalized = _normalize_grayscale(raw_image)
    gradient = _gradient_magnitude(normalized)
    local_variance = _local_variance_proxy(normalized)

    hotspot_threshold = _histogram_percentile(normalized, 0.95)
    hotspot = normalized.point(lambda pixel: 255 if pixel >= hotspot_threshold else 0)

    normalized_path = out_dir / "normalized.png"
    gradient_path = out_dir / "gradient_magnitude.png"
    local_variance_path = out_dir / "local_variance.png"
    hotspot_path = out_dir / "hotspot.png"
    global_path = out_dir / "global.json"
    manifest_path = out_dir / "manifest.json"

    normalized.save(normalized_path)
    gradient.save(gradient_path)
    local_variance.save(local_variance_path)
    hotspot.save(hotspot_path)

    stat = ImageStat.Stat(normalized)
    gradient_stat = ImageStat.Stat(gradient)
    hotspot_pixels = sum(1 for value in hotspot.getdata() if value > 0)
    total_pixels = normalized.size[0] * normalized.size[1]
    global_summary = {
        "source_path": str(ir_path),
        "width": normalized.size[0],
        "height": normalized.size[1],
        "mean_intensity": round(stat.mean[0], 6),
        "std_intensity": round(stat.stddev[0], 6),
        "hotspot_threshold": hotspot_threshold,
        "hotspot_ratio": round(hotspot_pixels / max(total_pixels, 1), 6),
        "hotspot_area_fraction": round(hotspot_pixels / max(total_pixels, 1), 6),
        "maximum_thermal_gradient": round(gradient.getextrema()[1], 6),
        "mean_hotspot_intensity": round(_masked_mean(normalized, hotspot), 6),
        "hotspot_compactness": round(_mask_compactness(hotspot), 6),
        "gradient_mean": round(gradient_stat.mean[0], 6),
    }
    global_path.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")

    artifacts = [
        build_descriptor_metadata(
            name="infrared_normalized",
            kind="spatial",
            path=normalized_path,
            channels=1,
            width=normalized.size[0],
            height=normalized.size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="infrared_gradient_magnitude",
            kind="spatial",
            path=gradient_path,
            channels=1,
            width=gradient.size[0],
            height=gradient.size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="infrared_local_variance",
            kind="spatial",
            path=local_variance_path,
            channels=1,
            width=local_variance.size[0],
            height=local_variance.size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="infrared_hotspot",
            kind="spatial",
            path=hotspot_path,
            channels=1,
            width=hotspot.size[0],
            height=hotspot.size[1],
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="infrared_global",
            kind="global",
            path=global_path,
            vector_length=7,
            dtype="json",
        ),
    ]
    write_descriptor_manifest(artifacts, manifest_path)

    return {
        "artifacts": artifacts,
        "manifest_path": manifest_path,
        "global_summary": global_summary,
    }
