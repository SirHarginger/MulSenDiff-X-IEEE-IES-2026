from pathlib import Path

import numpy as np

from src.preprocessing.pointcloud_projector import compute_category_bbox_stats, compute_pointcloud_feature_maps


def _write_grid_stl(
    path: Path,
    *,
    tx: float,
    ty: float,
    cell_size: float = 1.0,
    cells_x: int = 10,
    cells_y: int = 10,
    z: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["solid grid"]
    for ix in range(cells_x):
        for iy in range(cells_y):
            x0 = tx + ix * cell_size
            y0 = ty + iy * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            lines.extend(
                [
                    "facet normal 0 0 1",
                    "outer loop",
                    f"vertex {x0} {y0} {z}",
                    f"vertex {x1} {y0} {z}",
                    f"vertex {x0} {y1} {z}",
                    "endloop",
                    "endfacet",
                    "facet normal 0 0 1",
                    "outer loop",
                    f"vertex {x1} {y0} {z}",
                    f"vertex {x1} {y1} {z}",
                    f"vertex {x0} {y1} {z}",
                    "endloop",
                    "endfacet",
                ]
            )
    lines.append("endsolid grid")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_pointcloud_projection_centers_xy_per_sample(tmp_path: Path) -> None:
    sample_a = tmp_path / "a.stl"
    sample_b = tmp_path / "b.stl"
    _write_grid_stl(sample_a, tx=0.0, ty=0.0)
    _write_grid_stl(sample_b, tx=1000.0, ty=500.0)

    stats = compute_category_bbox_stats([sample_a, sample_b])
    projection_size = (64, 64)
    feature_maps = compute_pointcloud_feature_maps(
        sample_a,
        stats={**stats, "normal_deviation_mean": np.zeros((*projection_size, 3), dtype=np.float32)},
        projection_size=projection_size,
    )

    occupied_pixels = int(np.count_nonzero(feature_maps["density"]))
    assert occupied_pixels >= 100
    assert float(stats["x_bounds"][1] - stats["x_bounds"][0]) < 2.0
    assert float(stats["y_bounds"][1] - stats["y_bounds"][0]) < 2.0
