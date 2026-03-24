from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

from src.data_loader import SPATIAL_DESCRIPTOR_ORDER, SUPPORT_DESCRIPTOR_ORDER, iter_processed_sample_manifest


RECIPES = (
    "thermal_hotspot",
    "geometric_surface_defect",
    "cross_modal_inconsistency",
)


@dataclass(frozen=True)
class SyntheticBundleRecord:
    bundle_id: str
    recipe: str
    source_sample_dir: str
    source_sample_id: str
    category: str
    split: str
    output_dir: str
    rgb_path: str
    evidence_path: str
    provenance_path: str


def generate_synthetic_anomaly_bundles(
    *,
    samples_root: Path | str = "data/processed/samples",
    output_root: Path | str = "data/synthetic",
    recipes: Sequence[str] = RECIPES,
    limit: int = 0,
) -> Dict[str, Path]:
    samples_root = Path(samples_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in iter_processed_sample_manifest(samples_root)
        if row.split == "train" and not row.is_anomalous
    ]
    if limit:
        rows = rows[:limit]

    manifest_rows: List[Dict[str, str]] = []
    bundle_index: List[Dict[str, str]] = []
    for row in rows:
        sample_dir = Path(row.sample_dir)
        for recipe in recipes:
            bundle = _generate_bundle(sample_dir=sample_dir, recipe=recipe, output_root=output_root, row=row)
            manifest_rows.append(asdict(bundle))
            bundle_index.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "recipe": bundle.recipe,
                    "category": bundle.category,
                    "split": bundle.split,
                    "output_dir": bundle.output_dir,
                }
            )

    manifest_csv = output_root / "synthetic_manifest.csv"
    _write_manifest(manifest_csv, manifest_rows)
    index_json = output_root / "index.json"
    index_json.write_text(json.dumps(bundle_index, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest_csv": manifest_csv, "index_json": index_json}


def _generate_bundle(*, sample_dir: Path, recipe: str, output_root: Path, row) -> SyntheticBundleRecord:
    bundle_id = f"{row.category}__{row.sample_id}__{recipe}"
    output_dir = output_root / row.category / recipe / bundle_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(sample_dir / "rgb.png")
    descriptors = _load_gray_stack(sample_dir, SPATIAL_DESCRIPTOR_ORDER)
    support_maps = _load_gray_stack(sample_dir, SUPPORT_DESCRIPTOR_ORDER)
    mask, provenance = _build_mask(rgb.shape[1:], sample_id=row.sample_id, recipe=recipe)

    perturbed_descriptors, perturbed_support, descriptor_deltas = _apply_recipe(recipe, descriptors, support_maps, mask)
    synthetic_rgb = _render_synthetic_rgb(rgb, mask, recipe)

    _save_rgb(output_dir / "rgb.png", synthetic_rgb)
    _save_gray_stack(output_dir, SPATIAL_DESCRIPTOR_ORDER, perturbed_descriptors)
    _save_gray_stack(output_dir, SUPPORT_DESCRIPTOR_ORDER, perturbed_support)

    evidence = {
        "bundle_id": bundle_id,
        "recipe": recipe,
        "category": row.category,
        "source_sample_id": row.sample_id,
        "summary": _recipe_summary(recipe),
        "descriptor_deltas": descriptor_deltas,
        "region": provenance["region"],
        "expected_effect": provenance["expected_effect"],
    }
    evidence_path = output_dir / "evidence_package.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")

    provenance_payload = {
        "bundle_id": bundle_id,
        "recipe": recipe,
        "source_sample_dir": str(sample_dir),
        "seed": provenance["seed"],
        "region": provenance["region"],
        "expected_effect": provenance["expected_effect"],
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance_payload, indent=2, sort_keys=True), encoding="utf-8")

    return SyntheticBundleRecord(
        bundle_id=bundle_id,
        recipe=recipe,
        source_sample_dir=str(sample_dir),
        source_sample_id=row.sample_id,
        category=row.category,
        split=row.split,
        output_dir=str(output_dir),
        rgb_path=str(output_dir / "rgb.png"),
        evidence_path=str(evidence_path),
        provenance_path=str(provenance_path),
    )


def _build_mask(size: tuple[int, int], *, sample_id: str, recipe: str) -> tuple[torch.Tensor, Dict[str, object]]:
    height, width = size
    seed_value = int(hashlib.sha256(f"{sample_id}:{recipe}".encode("utf-8")).hexdigest()[:8], 16)
    generator = torch.Generator()
    generator.manual_seed(seed_value)

    center_x = int(torch.randint(low=width // 4, high=max(width // 4 + 1, (3 * width) // 4), size=(1,), generator=generator).item())
    center_y = int(torch.randint(low=height // 4, high=max(height // 4 + 1, (3 * height) // 4), size=(1,), generator=generator).item())
    sigma = max(min(height, width) // 8, 2)

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma * sigma))
    mask = mask.clamp(0.0, 1.0)
    region = {
        "centroid_xy": [center_x, center_y],
        "bbox_xyxy": [
            max(center_x - 2 * sigma, 0),
            max(center_y - 2 * sigma, 0),
            min(center_x + 2 * sigma, width - 1),
            min(center_y + 2 * sigma, height - 1),
        ],
    }
    return mask, {"seed": seed_value, "region": region, "expected_effect": _recipe_summary(recipe)}


def _apply_recipe(
    recipe: str,
    descriptors: Dict[str, torch.Tensor],
    support_maps: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
    descriptor_deltas: Dict[str, float] = {}
    updated_descriptors = {key: value.clone() for key, value in descriptors.items()}
    updated_support = {key: value.clone() for key, value in support_maps.items()}

    def bump(name: str, amount: float, *, support: bool = False) -> None:
        target = updated_support if support else updated_descriptors
        target[name] = (target[name] + amount * mask).clamp(0.0, 1.0)
        descriptor_deltas[name] = round(float((amount * mask).mean().item()), 6)

    if recipe == "thermal_hotspot":
        bump("ir_hotspot", 0.75)
        bump("ir_gradient", 0.45)
        bump("ir_variance", 0.35)
        bump("cross_agreement", 0.25)
        bump("cross_ir_support", 0.5, support=True)
    elif recipe == "geometric_surface_defect":
        bump("pc_curvature", 0.55)
        bump("pc_roughness", 0.65)
        bump("pc_normal_deviation", 0.45)
        bump("cross_agreement", 0.2)
        bump("cross_geo_support", 0.5, support=True)
    elif recipe == "cross_modal_inconsistency":
        bump("ir_hotspot", 0.45)
        bump("cross_agreement", -0.25)
        bump("cross_inconsistency", 0.7, support=True)
        bump("cross_ir_support", 0.25, support=True)
    else:
        raise ValueError(f"Unsupported recipe={recipe!r}")

    return updated_descriptors, updated_support, descriptor_deltas


def _render_synthetic_rgb(rgb: torch.Tensor, mask: torch.Tensor, recipe: str) -> torch.Tensor:
    rendered = rgb.clone()
    if recipe == "thermal_hotspot":
        rendered[0] = (rendered[0] + 0.35 * mask).clamp(0.0, 1.0)
        rendered[1] = (rendered[1] + 0.15 * mask).clamp(0.0, 1.0)
    elif recipe == "geometric_surface_defect":
        rendered[0] = (rendered[0] - 0.20 * mask).clamp(0.0, 1.0)
        rendered[1] = (rendered[1] - 0.15 * mask).clamp(0.0, 1.0)
        rendered[2] = (rendered[2] - 0.10 * mask).clamp(0.0, 1.0)
    else:
        rendered[2] = (rendered[2] + 0.30 * mask).clamp(0.0, 1.0)
        rendered[0] = (rendered[0] - 0.10 * mask).clamp(0.0, 1.0)
    return rendered


def _load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _load_gray_stack(sample_dir: Path, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    stack: Dict[str, torch.Tensor] = {}
    for name in names:
        image = Image.open(sample_dir / f"{name}.png").convert("L")
        stack[name] = torch.from_numpy(np.asarray(image, dtype=np.float32)) / 255.0
    return stack


def _save_rgb(path: Path, rgb: torch.Tensor) -> None:
    array = (rgb.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path)


def _save_gray_stack(output_dir: Path, names: Sequence[str], stack: Dict[str, torch.Tensor]) -> None:
    for name in names:
        array = (stack[name].clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
        Image.fromarray(array, mode="L").save(output_dir / f"{name}.png")


def _write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else list(SyntheticBundleRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _recipe_summary(recipe: str) -> str:
    if recipe == "thermal_hotspot":
        return "Localized thermal anomaly with supporting hotspot and gradient cues."
    if recipe == "geometric_surface_defect":
        return "Localized structural disturbance with elevated curvature and roughness."
    return "Cross-modal disagreement where thermal suspicion is not matched by geometry."
