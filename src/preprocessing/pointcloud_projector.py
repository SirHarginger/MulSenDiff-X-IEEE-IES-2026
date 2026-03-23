from __future__ import annotations

from functools import lru_cache
import json
import struct
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
except ModuleNotFoundError:  # pragma: no cover
    distance_transform_edt = None
    cKDTree = None


PC_BBOX_STATS_JSON = "pc_bbox_stats.json"


# ─────────────────────────────────────────────────────────────────────────────
# STL parsing
# ─────────────────────────────────────────────────────────────────────────────

def _is_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as fh:
        fh.read(80)
        triangle_count = struct.unpack("<I", fh.read(4))[0]
    return 84 + triangle_count * 50 == size


def parse_stl_mesh(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a binary or ASCII STL file.

    Returns
    -------
    vertices      float32 (N, 3)
    faces         int32   (F, 3)   — indices into vertices
    face_normals  float32 (F, 3)
    """
    return _parse_stl_mesh_cached(str(Path(path).expanduser().resolve()))


@lru_cache(maxsize=None)
def _parse_stl_mesh_cached(path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_str)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_normals_list: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    def _get_idx(v: tuple[float, float, float]) -> int:
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append(key)
        return vertex_index[key]

    if _is_binary_stl(path):
        with path.open("rb") as fh:
            fh.read(80)
            n_tri = struct.unpack("<I", fh.read(4))[0]
            for _ in range(n_tri):
                chunk = fh.read(50)
                if len(chunk) < 50:
                    break
                vals = struct.unpack("<12fH", chunk)
                face_normals_list.append((vals[0], vals[1], vals[2]))
                tri = [
                    (vals[3], vals[4], vals[5]),
                    (vals[6], vals[7], vals[8]),
                    (vals[9], vals[10], vals[11]),
                ]
                faces.append(tuple(_get_idx(v) for v in tri))
    else:
        current_normal = (0.0, 0.0, 1.0)
        pending: list[tuple[float, float, float]] = []
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            elif line.startswith("vertex"):
                parts = line.split()
                pending.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(pending) == 3:
                    faces.append(tuple(_get_idx(v) for v in pending))
                    face_normals_list.append(current_normal)
                    pending = []

    verts = np.asarray(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32)
    f_arr = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    fn_arr = np.asarray(face_normals_list, dtype=np.float32) if face_normals_list else np.zeros((0, 3), dtype=np.float32)
    for array in (verts, f_arr, fn_arr):
        array.setflags(write=False)
    return verts, f_arr, fn_arr


# ─────────────────────────────────────────────────────────────────────────────
# Vertex operations — all vectorised, no Python loops over vertices or faces
# ─────────────────────────────────────────────────────────────────────────────

def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
) -> np.ndarray:
    """
    Area-weighted vertex normals.  Fully vectorised — no Python face loop.

    Parameters
    ----------
    vertices      float32 (N, 3)
    faces         int32   (F, 3)
    face_normals  float32 (F, 3)  stored normals from STL

    Returns
    -------
    float32 (N, 3)  unit vertex normals
    """
    if vertices.size == 0 or faces.size == 0:
        return np.zeros_like(vertices, dtype=np.float32)

    v0 = vertices[faces[:, 0]]          # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    areas = np.linalg.norm(cross, axis=1, keepdims=True) / 2.0  # (F, 1)

    # prefer stored STL normals; fall back to computed cross product
    fn = face_normals.copy()
    fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
    bad = fn_norms[:, 0] <= 1e-8
    fn[bad] = cross[bad]
    fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
    still_bad = fn_norms[:, 0] <= 1e-8
    fn[still_bad] = [0.0, 0.0, 1.0]
    fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.clip(fn_norms, 1e-8, None)   # (F, 3) unit normals

    weighted = fn * np.clip(areas, 1e-8, None)  # (F, 3)

    accumulated = np.zeros_like(vertices, dtype=np.float32)  # (N, 3)
    np.add.at(accumulated, faces[:, 0], weighted)
    np.add.at(accumulated, faces[:, 1], weighted)
    np.add.at(accumulated, faces[:, 2], weighted)

    norms = np.linalg.norm(accumulated, axis=1, keepdims=True)
    return (accumulated / np.clip(norms, 1e-8, None)).astype(np.float32)


def build_knn_indices(vertices: np.ndarray, *, k: int = 12) -> np.ndarray:
    """
    K-nearest-neighbour indices using scipy cKDTree (runs in C).

    Parameters
    ----------
    vertices  float32 (N, 3)
    k         number of neighbours

    Returns
    -------
    int32 (N, k)
    """
    n = int(vertices.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.int32)
    if n == 1:
        return np.zeros((1, 0), dtype=np.int32)

    k_actual = min(k, n - 1)
    if cKDTree is None:
        indices = np.zeros((n, k_actual), dtype=np.int32)
        for index in range(n):
            distances = np.sum((vertices - vertices[index]) ** 2, axis=1)
            kth = min(k_actual, n - 1)
            nearest = np.argpartition(distances, kth)[: k_actual + 1]
            nearest = nearest[nearest != index][:k_actual]
            if nearest.shape[0] < k_actual:
                pad_value = nearest[-1] if nearest.shape[0] else index
                nearest = np.pad(nearest, (0, k_actual - nearest.shape[0]), constant_values=pad_value)
            indices[index] = nearest
        return indices

    tree = cKDTree(vertices)
    # query k+1 because the closest point to each vertex is itself
    _, indices = tree.query(vertices, k=k_actual + 1, workers=-1)
    # drop the self-match (always column 0) and return the rest
    return indices[:, 1:].astype(np.int32)


def _compute_curvature(
    vertex_normals: np.ndarray,
    knn_indices: np.ndarray,
) -> np.ndarray:
    """
    Per-vertex curvature as mean angular deviation among k neighbours.
    Fully vectorised — no Python vertex loop.

    Returns float32 (N,) in radians.
    """
    if vertex_normals.size == 0 or knn_indices.size == 0:
        return np.zeros((vertex_normals.shape[0],), dtype=np.float32)

    neighbor_normals = vertex_normals[knn_indices]              # (N, k, 3)
    reference = vertex_normals[:, np.newaxis, :]                # (N, 1, 3)
    dots = np.clip(
        np.sum(neighbor_normals * reference, axis=2), -1.0, 1.0
    )                                                           # (N, k)
    return np.mean(np.arccos(dots), axis=1).astype(np.float32) # (N,)


def _compute_roughness(
    vertices: np.ndarray,
    knn_indices: np.ndarray,
) -> np.ndarray:
    """
    Per-vertex roughness as RMS distance to locally fitted plane.
    Uses batch SVD — no Python vertex loop.

    Returns float32 (N,).
    """
    if vertices.size == 0 or knn_indices.size == 0:
        return np.zeros((vertices.shape[0],), dtype=np.float32)

    n, k = knn_indices.shape
    neighbor_verts = vertices[knn_indices]                      # (N, k, 3)
    self_verts = vertices[:, np.newaxis, :]                     # (N, 1, 3)
    all_points = np.concatenate([self_verts, neighbor_verts], axis=1)  # (N, k+1, 3)

    centroid = all_points.mean(axis=1, keepdims=True)           # (N, 1, 3)
    centered = all_points - centroid                            # (N, k+1, 3)

    # batch SVD: right singular vectors give the plane normal (last row of vh)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)     # vh: (N, 3, 3)
    plane_normals = vh[:, -1, :]                                # (N, 3)

    # signed distances of all k+1 points from their plane
    distances = np.sum(
        centered * plane_normals[:, np.newaxis, :], axis=2
    )                                                           # (N, k+1)
    return np.sqrt(np.mean(distances ** 2, axis=1)).astype(np.float32)  # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Projection utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_vertices(
    vertices: np.ndarray,
    *,
    center: np.ndarray,
    diagonal: float,
) -> np.ndarray:
    return (vertices - center.reshape(1, 3)) / max(float(diagonal), 1e-6)


def _normalize_vertices_for_projection(
    vertices: np.ndarray,
    *,
    mean_center: np.ndarray,
    mean_diagonal: float,
) -> np.ndarray:
    normalized = normalize_vertices(vertices, center=mean_center, diagonal=mean_diagonal)
    if vertices.size == 0:
        return normalized

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    sample_center_xy = ((lo + hi) / 2.0)[:2]
    category_center_xy = np.asarray(mean_center[:2], dtype=np.float32)
    xy_offset = (sample_center_xy - category_center_xy) / max(float(mean_diagonal), 1e-6)
    normalized[:, :2] -= xy_offset.reshape(1, 2)
    return normalized.astype(np.float32, copy=False)


def _vertices_to_grid_coords(
    normalized: np.ndarray,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map normalised vertex XY to integer grid pixel coordinates.
    Fully vectorised — returns (px, py) integer arrays of shape (N,).
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    px = np.clip(
        np.round(
            (normalized[:, 0] - x_min) / max(x_max - x_min, 1e-8) * (width - 1)
        ),
        0, width - 1,
    ).astype(np.int32)

    py = np.clip(
        np.round(
            (normalized[:, 1] - y_min) / max(y_max - y_min, 1e-8) * (height - 1)
        ),
        0, height - 1,
    ).astype(np.int32)

    return px, py


def _fill_nan_nearest(array: np.ndarray) -> np.ndarray:
    """
    Fill NaN cells with the value of the nearest non-NaN cell.
    Uses scipy distance_transform_edt — single pass, no iteration.
    """
    mask = np.isnan(array)
    if not np.any(mask):
        return array.astype(np.float32)
    if distance_transform_edt is None:
        return _fill_nan_nearest_fallback(array)
    indices = distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    filled = array[tuple(indices)]
    filled[np.isnan(filled)] = 0.0
    return filled.astype(np.float32)


def _fill_nan_nearest_fallback(array: np.ndarray) -> np.ndarray:
    filled = array.copy()
    valid = ~np.isnan(filled)
    if not np.any(valid):
        return np.zeros_like(filled, dtype=np.float32)

    # Iteratively propagate neighboring valid values until the grid is filled.
    while not np.all(valid):
        previous_valid = valid.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_values = np.roll(filled, shift=(dy, dx), axis=(0, 1))
            shifted_valid = np.roll(valid, shift=(dy, dx), axis=(0, 1))
            if dy == -1:
                shifted_valid[-1, :] = False
            elif dy == 1:
                shifted_valid[0, :] = False
            if dx == -1:
                shifted_valid[:, -1] = False
            elif dx == 1:
                shifted_valid[:, 0] = False

            update_mask = ~valid & shifted_valid
            filled[update_mask] = shifted_values[update_mask]
            valid[update_mask] = True
        if np.array_equal(valid, previous_valid):
            filled[~valid] = 0.0
            break

    filled[np.isnan(filled)] = 0.0
    return filled.astype(np.float32)


def _normalize_signed_map(
    array: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    if maximum <= minimum + 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(
        (array - minimum) / (maximum - minimum), 0.0, 1.0
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Category-level statistics (run once per category over training split)
# ─────────────────────────────────────────────────────────────────────────────

def compute_category_bbox_stats(
    pointcloud_paths: Sequence[Path | str],
) -> Dict[str, object]:
    if not pointcloud_paths:
        raise ValueError("At least one point-cloud path is required.")

    centers: list[np.ndarray] = []
    diagonals: list[float] = []

    for path in pointcloud_paths:
        verts, _, _ = parse_stl_mesh(path)
        if verts.size == 0:
            continue
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        centers.append((lo + hi) / 2.0)
        diagonals.append(float(np.linalg.norm(hi - lo)))

    if not centers:
        raise ValueError("No vertices parsed from training point clouds.")

    mean_center = np.stack(centers).mean(axis=0)
    mean_diagonal = max(float(np.mean(diagonals)), 1e-6)

    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")

    for path in pointcloud_paths:
        verts, _, _ = parse_stl_mesh(path)
        if verts.size == 0:
            continue
        n = _normalize_vertices_for_projection(verts, mean_center=mean_center, mean_diagonal=mean_diagonal)
        x_min, x_max = min(x_min, float(n[:, 0].min())), max(x_max, float(n[:, 0].max()))
        y_min, y_max = min(y_min, float(n[:, 1].min())), max(y_max, float(n[:, 1].max()))
        z_min, z_max = min(z_min, float(n[:, 2].min())), max(z_max, float(n[:, 2].max()))

    return {
        "mean_center": mean_center.astype(np.float32),
        "mean_diagonal": mean_diagonal,
        "x_bounds": [x_min, x_max],
        "y_bounds": [y_min, y_max],
        "z_bounds": [z_min, z_max],
    }


def save_pointcloud_category_stats(
    stats: Dict[str, object],
    out_dir: Path | str,
    *,
    depth_mean: np.ndarray,
    roughness_mean: np.ndarray,
    curvature_mean: np.ndarray,
    normal_deviation_mean: np.ndarray,
    density_max: float,
    depth_residual_range: tuple[float, float],
    roughness_residual_max: float,
    curvature_residual_max: float,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "depth_mean.npy", depth_mean.astype(np.float32))
    np.save(out_dir / "roughness_mean.npy", roughness_mean.astype(np.float32))
    np.save(out_dir / "curvature_mean.npy", curvature_mean.astype(np.float32))
    np.save(out_dir / "normal_deviation_mean.npy", normal_deviation_mean.astype(np.float32))

    payload = {
        "mean_center": np.asarray(stats["mean_center"], dtype=np.float32).tolist(),
        "mean_diagonal": float(stats["mean_diagonal"]),
        "x_bounds": list(stats["x_bounds"]),
        "y_bounds": list(stats["y_bounds"]),
        "z_bounds": list(stats["z_bounds"]),
        "density_max": float(max(density_max, 1.0)),
        "depth_residual_min": float(depth_residual_range[0]),
        "depth_residual_max": float(depth_residual_range[1]),
        "roughness_residual_max": float(max(roughness_residual_max, 1e-6)),
        "curvature_residual_max": float(max(curvature_residual_max, 1e-6)),
    }
    (out_dir / PC_BBOX_STATS_JSON).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_pointcloud_category_stats(
    stats_dir: Path | str,
) -> Dict[str, object]:
    stats_dir = Path(stats_dir)
    payload = json.loads((stats_dir / PC_BBOX_STATS_JSON).read_text(encoding="utf-8"))
    return {
        **payload,
        "mean_center": np.asarray(payload["mean_center"], dtype=np.float32),
        "depth_mean": np.load(stats_dir / "depth_mean.npy"),
        "roughness_mean": np.load(stats_dir / "roughness_mean.npy"),
        "curvature_mean": np.load(stats_dir / "curvature_mean.npy"),
        "normal_deviation_mean": np.load(stats_dir / "normal_deviation_mean.npy"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core feature map computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pointcloud_feature_maps(
    pointcloud_path: Path | str,
    *,
    stats: Dict[str, object],
    projection_size: tuple[int, int] = (256, 256),
    k: int = 12,
) -> Dict[str, np.ndarray]:
    """
    Compute all raw point-cloud feature maps for one STL file.

    All expensive operations (KNN, normals, curvature, roughness, projection)
    are vectorised.  No Python loops over vertices or faces.

    Returns
    -------
    depth            float32 (H, W)   — NaN-filled mean Z projection
    density          float32 (H, W)   — vertex count per cell
    curvature        float32 (H, W)   — max curvature per cell (radians)
    roughness        float32 (H, W)   — mean roughness per cell
    normal_map       float32 (H, W, 3)— mean unit normal per cell
    normal_deviation float32 (H, W)   — angular deviation from expected (0-1)
    valid_mask       float32 (H, W)   — 1 where density > 0
    """
    width, height = projection_size
    _zero = lambda: np.zeros((height, width), dtype=np.float32)

    verts, faces, face_normals = parse_stl_mesh(pointcloud_path)

    if verts.size == 0:
        return {
            "depth": _zero(),
            "density": _zero(),
            "curvature": _zero(),
            "roughness": _zero(),
            "normal_map": np.zeros((height, width, 3), dtype=np.float32),
            "normal_deviation": _zero(),
            "valid_mask": _zero(),
        }

    # ── normalise to per-category coordinate space ───────────────────────────
    norm_verts = _normalize_vertices_for_projection(
        verts,
        mean_center=np.asarray(stats["mean_center"], dtype=np.float32),
        mean_diagonal=float(stats["mean_diagonal"]),
    )

    # ── vertex-level features ─────────────────────────────────────────────────
    vertex_normals = compute_vertex_normals(verts, faces, face_normals)
    knn_idx = build_knn_indices(norm_verts, k=k)
    curvature = _compute_curvature(vertex_normals, knn_idx)   # (N,) radians
    roughness = _compute_roughness(norm_verts, knn_idx)        # (N,)

    # ── vectorised projection ─────────────────────────────────────────────────
    x_bounds = tuple(float(v) for v in stats["x_bounds"])
    y_bounds = tuple(float(v) for v in stats["y_bounds"])
    px, py = _vertices_to_grid_coords(
        norm_verts,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        width=width,
        height=height,
    )

    # depth — mean Z per cell
    depth_sum = np.zeros((height, width), dtype=np.float64)
    density   = np.zeros((height, width), dtype=np.float32)
    np.add.at(depth_sum, (py, px), norm_verts[:, 2].astype(np.float64))
    np.add.at(density,   (py, px), 1.0)

    # curvature — max per cell
    curvature_grid = np.zeros((height, width), dtype=np.float32)
    np.maximum.at(curvature_grid, (py, px), curvature)

    # roughness — mean per cell
    roughness_sum   = np.zeros((height, width), dtype=np.float64)
    roughness_count = np.zeros((height, width), dtype=np.float32)
    np.add.at(roughness_sum,   (py, px), roughness.astype(np.float64))
    np.add.at(roughness_count, (py, px), 1.0)

    # normals — mean per cell then renormalise
    normal_sum = np.zeros((height, width, 3), dtype=np.float64)
    np.add.at(normal_sum, (py, px), vertex_normals.astype(np.float64))

    # ── assemble grid arrays ─────────────────────────────────────────────────
    valid = density > 0

    depth_grid = np.full((height, width), np.nan, dtype=np.float32)
    depth_grid[valid] = (depth_sum[valid] / density[valid]).astype(np.float32)
    depth_grid = _fill_nan_nearest(depth_grid)

    roughness_grid = np.zeros((height, width), dtype=np.float32)
    roughness_grid[valid] = (
        roughness_sum[valid] / np.clip(roughness_count[valid], 1.0, None)
    ).astype(np.float32)

    curvature_grid[~valid] = 0.0
    roughness_grid[~valid] = 0.0

    normal_map = np.zeros((height, width, 3), dtype=np.float32)
    if np.any(valid):
        nm = (normal_sum[valid] / density[valid][:, None]).astype(np.float32)
        nm_norms = np.linalg.norm(nm, axis=1, keepdims=True)
        normal_map[valid] = nm / np.clip(nm_norms, 1e-8, None)

    # ── angular deviation from per-category expected normals ─────────────────
    expected = np.asarray(stats["normal_deviation_mean"], dtype=np.float32)
    dots = np.clip(np.sum(normal_map * expected, axis=2), -1.0, 1.0)
    deviation = np.degrees(np.arccos(dots)).astype(np.float32)
    deviation[~valid] = 0.0
    normal_deviation = np.clip(deviation / 180.0, 0.0, 1.0).astype(np.float32)

    return {
        "depth": depth_grid,
        "density": density,
        "curvature": curvature_grid,
        "roughness": roughness_grid,
        "normal_map": normal_map,
        "normal_deviation": normal_deviation,
        "valid_mask": valid.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample descriptor generation (residual normalisation)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pointcloud_sample_descriptors(
    pointcloud_path: Path | str,
    *,
    stats: Dict[str, object],
    projection_size: tuple[int, int] = (256, 256),
) -> Dict[str, object]:
    """
    Produce residual descriptor maps and global statistics for one sample.

    All maps are float32 normalised to [0, 1].

    Returns keys
    ------------
    pc_depth, pc_density, pc_curvature, pc_roughness,
    pc_normal_deviation  — each float32 (H, W)
    global               — dict with three scalar entries
    """
    fm = compute_pointcloud_feature_maps(
        pointcloud_path, stats=stats, projection_size=projection_size
    )

    # depth residual: signed, clipped to training range, then 0-1
    depth_res = fm["depth"] - np.asarray(stats["depth_mean"], dtype=np.float32)
    depth_norm = _normalize_signed_map(
        depth_res,
        minimum=float(stats["depth_residual_min"]),
        maximum=float(stats["depth_residual_max"]),
    )

    # density: absolute (not a residual)
    density_norm = np.clip(
        fm["density"] / max(float(stats["density_max"]), 1.0), 0.0, 1.0
    ).astype(np.float32)

    # curvature residual: clip negatives, normalise by training max
    curv_res = np.clip(
        fm["curvature"] - np.asarray(stats["curvature_mean"], dtype=np.float32),
        0.0, None,
    )
    curv_norm = np.clip(
        curv_res / max(float(stats["curvature_residual_max"]), 1e-6),
        0.0, 1.0,
    ).astype(np.float32)

    # roughness residual: clip negatives, normalise by training max
    rough_res = np.clip(
        fm["roughness"] - np.asarray(stats["roughness_mean"], dtype=np.float32),
        0.0, None,
    )
    rough_norm = np.clip(
        rough_res / max(float(stats["roughness_residual_max"]), 1e-6),
        0.0, 1.0,
    ).astype(np.float32)

    # normal deviation already 0-1
    nd_norm = fm["normal_deviation"].astype(np.float32)

    # global statistics — computed only over valid (non-zero density) cells
    valid = fm["valid_mask"] > 0
    curv_vals = curv_norm[valid]
    rough_vals = rough_norm[valid]
    nd_vals = nd_norm[valid]

    return {
        "pc_depth": depth_norm,
        "pc_density": density_norm,
        "pc_curvature": curv_norm,
        "pc_roughness": rough_norm,
        "pc_normal_deviation": nd_norm,
        "global": {
            "pc_mean_curvature": round(
                float(curv_vals.mean()) if curv_vals.size else 0.0, 6
            ),
            "pc_max_roughness": round(
                float(rough_vals.max()) if rough_vals.size else 0.0, 6
            ),
            "pc_p95_normal_deviation": round(
                float(np.percentile(nd_vals, 95)) if nd_vals.size else 0.0, 6
            ),
        },
    }
