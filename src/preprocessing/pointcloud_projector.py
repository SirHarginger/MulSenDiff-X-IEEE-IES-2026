from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageStat

from src.preprocessing.crossmodal_descriptors import build_descriptor_metadata, write_descriptor_manifest


def _is_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as handle:
        handle.read(80)
        triangle_count = struct.unpack("<I", handle.read(4))[0]
    return 84 + triangle_count * 50 == size


def iter_stl_vertices(path: Path | str) -> Iterator[Tuple[float, float, float]]:
    path = Path(path)
    if _is_binary_stl(path):
        with path.open("rb") as handle:
            handle.read(80)
            triangle_count = struct.unpack("<I", handle.read(4))[0]
            for _ in range(triangle_count):
                chunk = handle.read(50)
                if len(chunk) < 50:
                    break
                values = struct.unpack("<12fH", chunk)
                yield values[3], values[4], values[5]
                yield values[6], values[7], values[8]
                yield values[9], values[10], values[11]
    else:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("vertex"):
                _, x, y, z = line.split()
                yield float(x), float(y), float(z)


def _scan_bounds(path: Path) -> Dict[str, float]:
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    count = 0

    for x, y, z in iter_stl_vertices(path):
        count += 1
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    if count == 0:
        return {
            "count": 0,
            "min_x": 0.0,
            "max_x": 0.0,
            "min_y": 0.0,
            "max_y": 0.0,
            "min_z": 0.0,
            "max_z": 0.0,
        }

    return {
        "count": count,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
    }


def _scale(value: float, minimum: float, maximum: float, size: int) -> int:
    if maximum <= minimum:
        return 0
    ratio = (value - minimum) / (maximum - minimum)
    return max(0, min(size - 1, int(ratio * (size - 1))))


def _gradient_magnitude(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    dx = ImageChops.difference(image, ImageChops.offset(image, 1, 0))
    dy = ImageChops.difference(image, ImageChops.offset(image, 0, 1))
    return ImageChops.lighter(dx, dy)


def generate_pointcloud_descriptor_bundle(
    pointcloud_path: Path | str,
    out_dir: Path | str,
    *,
    projection_size: Tuple[int, int] = (256, 256),
) -> Dict[str, object]:
    pointcloud_path = Path(pointcloud_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = _scan_bounds(pointcloud_path)
    width, height = projection_size

    depth_by_pixel: Dict[Tuple[int, int], int] = {}
    density_by_pixel: Dict[Tuple[int, int], int] = {}

    if bounds["count"] > 0:
        for x, y, z in iter_stl_vertices(pointcloud_path):
            px = _scale(x, bounds["min_x"], bounds["max_x"], width)
            py = _scale(y, bounds["min_y"], bounds["max_y"], height)
            pz = _scale(z, bounds["min_z"], bounds["max_z"], 256)
            key = (px, py)
            depth_by_pixel[key] = max(depth_by_pixel.get(key, 0), pz)
            density_by_pixel[key] = density_by_pixel.get(key, 0) + 1

    depth_image = Image.new("L", projection_size, 0)
    density_image = Image.new("L", projection_size, 0)
    max_density = max(density_by_pixel.values(), default=1)
    for key, depth in depth_by_pixel.items():
        depth_image.putpixel(key, depth)
    for key, count in density_by_pixel.items():
        density_value = int((count / max_density) * 255) if max_density else 0
        density_image.putpixel(key, density_value)

    smoothed_depth = depth_image.filter(ImageFilter.BoxBlur(2))
    curvature_map = _gradient_magnitude(smoothed_depth)
    roughness_map = ImageChops.difference(depth_image, smoothed_depth)
    normal_deviation_map = _gradient_magnitude(roughness_map)

    depth_path = out_dir / "depth_topdown.png"
    density_path = out_dir / "density_topdown.png"
    curvature_path = out_dir / "curvature.png"
    roughness_path = out_dir / "roughness.png"
    normal_deviation_path = out_dir / "normal_deviation.png"
    global_path = out_dir / "global.json"
    manifest_path = out_dir / "manifest.json"

    depth_image.save(depth_path)
    density_image.save(density_path)
    curvature_map.save(curvature_path)
    roughness_map.save(roughness_path)
    normal_deviation_map.save(normal_deviation_path)

    occupied_pixels = len(depth_by_pixel)
    total_pixels = width * height
    curvature_stat = ImageStat.Stat(curvature_map)
    roughness_extrema = roughness_map.getextrema()
    deviation_histogram = normal_deviation_map.histogram()
    deviation_target = sum(deviation_histogram) * 0.95
    deviation_cumulative = 0
    deviation_p95 = 255
    for value, count in enumerate(deviation_histogram):
        deviation_cumulative += count
        if deviation_cumulative >= deviation_target:
            deviation_p95 = value
            break
    global_summary = {
        "source_path": str(pointcloud_path),
        "vertex_count": bounds["count"],
        "x_range": round(bounds["max_x"] - bounds["min_x"], 6),
        "y_range": round(bounds["max_y"] - bounds["min_y"], 6),
        "z_range": round(bounds["max_z"] - bounds["min_z"], 6),
        "projection_width": width,
        "projection_height": height,
        "coverage_ratio": round(occupied_pixels / max(total_pixels, 1), 6),
        "mean_pointcloud_curvature": round(curvature_stat.mean[0], 6),
        "maximum_pointcloud_roughness": round(roughness_extrema[1], 6),
        "p95_normal_deviation": round(deviation_p95, 6),
    }
    global_path.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")

    artifacts = [
        build_descriptor_metadata(
            name="pointcloud_depth_topdown",
            kind="spatial",
            path=depth_path,
            channels=1,
            width=width,
            height=height,
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="pointcloud_density_topdown",
            kind="spatial",
            path=density_path,
            channels=1,
            width=width,
            height=height,
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="pointcloud_curvature",
            kind="spatial",
            path=curvature_path,
            channels=1,
            width=width,
            height=height,
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="pointcloud_roughness",
            kind="spatial",
            path=roughness_path,
            channels=1,
            width=width,
            height=height,
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="pointcloud_normal_deviation",
            kind="spatial",
            path=normal_deviation_path,
            channels=1,
            width=width,
            height=height,
            dtype="uint8",
        ),
        build_descriptor_metadata(
            name="pointcloud_global",
            kind="global",
            path=global_path,
            vector_length=8,
            dtype="json",
        ),
    ]
    write_descriptor_manifest(artifacts, manifest_path)

    return {
        "artifacts": artifacts,
        "manifest_path": manifest_path,
        "global_summary": global_summary,
    }
