# Descriptor Pipeline

## Purpose

MulSenDiff-X does not feed raw infrared images or raw point clouds directly into the detector as side channels. We first convert them into deterministic auxiliary descriptors that are:

- category-normalized from `train/good` data
- cached under `data/processed/samples/`
- aligned to the same 2D grid used by RGB
- split into descriptor maps, support maps, and a compact global vector

These descriptors are the multimodal conditioning signals consumed by training, evaluation, and the app.

## Where The Descriptors Come From

The full preprocessing path is implemented in:

- [src/preprocessing/descriptor_pipeline.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/descriptor_pipeline.py)
- [src/preprocessing/ir_descriptors.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/ir_descriptors.py)
- [src/preprocessing/pointcloud_projector.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/pointcloud_projector.py)
- [src/preprocessing/crossmodal_descriptors.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/crossmodal_descriptors.py)

The pipeline runs in two stages for each category:

1. Build category statistics from the training split.
2. Materialize one processed sample folder per record using those frozen category statistics.

The category statistics are computed before any per-sample descriptors are written, so the descriptors used at inference time are always normalized against category-level training data rather than recomputed from the test sample itself.

## Per-Sample Output Folder

Each processed sample is stored as:

`data/processed/samples/<category>__<split>__<defect_label>__<sample_id>/`

The folder contains:

- `rgb.png`
- `ir_normalized.png`
- `ir_gradient.png`
- `ir_variance.png`
- `ir_hotspot.png`
- `pc_depth.png`
- `pc_density.png`
- `pc_curvature.png`
- `pc_roughness.png`
- `pc_normal_deviation.png`
- `cross_ir_support.png`
- `cross_geo_support.png`
- `cross_agreement.png`
- `cross_agreement_preview.png`
- `cross_inconsistency.png`
- `global.json`
- `meta.json`
- `gt_mask.png` for anomalous test samples when an RGB mask exists

`meta.json` records dataset provenance such as category, split, defect label, sample id, and the original RGB / IR / STL source paths.

## Exact Model-Facing Descriptor Layout

The model consumes three descriptor groups, defined in [src/data_loader.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/data_loader.py).

### Spatial Descriptor Maps `M`

These channels are stacked in this exact order:

1. `ir_hotspot`
2. `ir_gradient`
3. `ir_variance`
4. `pc_curvature`
5. `pc_roughness`
6. `pc_normal_deviation`
7. `cross_agreement`

These are the strongest localized anomaly cues and are treated as the main descriptor tensor.

### Support Maps `S`

These channels are stacked in this exact order:

1. `ir_normalized`
2. `pc_depth`
3. `pc_density`
4. `cross_ir_support`
5. `cross_geo_support`
6. `cross_inconsistency`

These channels provide context, object support, and cross-modal alignment signals.

### Global Descriptor Vector `g`

The global vector is read from `global.json` in this exact order:

1. `ir_hotspot_area_fraction`
2. `ir_max_gradient`
3. `ir_mean_hotspot_intensity`
4. `ir_hotspot_compactness`
5. `pc_mean_curvature`
6. `pc_max_roughness`
7. `pc_p95_normal_deviation`
8. `cross_overlap_score`
9. `cross_centroid_distance`
10. `cross_inconsistency_score`

Only these ten fields are used as the learned global descriptor vector, even though `global.json` also carries some additional cross-modal alignment metadata.

## Infrared Descriptors

Infrared descriptors are generated in [src/preprocessing/ir_descriptors.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/ir_descriptors.py).

### Category Statistics

For each category, the pipeline loads all training IR images, converts them to grayscale, resizes them to the target size, and computes:

- a category-wide `global_min`
- a category-wide `global_max`
- `ir_mean`
- `ir_std`
- `ir_gradient_mean`
- `ir_variance_mean`
- `gradient_residual_max`
- `variance_residual_max`

`ir_std` is regularized with a small floor so the hotspot z-score does not explode in low-variance regions.

### Per-Sample IR Maps

Each sample IR image is resized and normalized with the category-level `global_min` and `global_max`, then the following maps are created:

- `ir_normalized`
  - the category-normalized grayscale IR image in `[0, 1]`

- `ir_gradient`
  - Gaussian-smoothed Sobel gradient magnitude
  - residual above the category `ir_gradient_mean`
  - clipped to positive values only
  - normalized by `gradient_residual_max`

- `ir_variance`
  - local variance over a `9 x 9` window
  - computed on `(ir_normalized - ir_mean)`
  - residual above the category `ir_variance_mean`
  - clipped to positive values only
  - normalized by `variance_residual_max`

- `ir_hotspot`
  - z-score map `(ir_normalized - ir_mean) / ir_std`
  - only pixels with `z >= 2.0` are kept
  - z-scores are clipped to `[0, 5]`
  - normalized by dividing by `5`

### IR Global Features

The IR branch contributes four global statistics:

- `ir_hotspot_area_fraction`
  - fraction of pixels where hotspot response is nonzero

- `ir_max_gradient`
  - maximum value in the normalized gradient residual map

- `ir_mean_hotspot_intensity`
  - mean clipped hotspot z-score over hotspot pixels

- `ir_hotspot_compactness`
  - binary hotspot compactness computed from area and perimeter
  - uses the standard `4 * pi * area / perimeter^2` form

## Point-Cloud Descriptors

Point-cloud descriptors are generated in [src/preprocessing/pointcloud_projector.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/pointcloud_projector.py).

### STL Parsing And Vertex Features

The implementation accepts both binary and ASCII STL files. For each mesh it:

- parses vertices, faces, and face normals
- computes area-weighted vertex normals
- builds a `k = 12` nearest-neighbor graph
- computes per-vertex curvature as mean angular deviation of neighboring normals
- computes per-vertex roughness as RMS distance to a locally fitted plane

### Category Geometry Statistics

Each category first gets a frozen geometric coordinate frame from its training point clouds:

- `mean_center`
- `mean_diagonal`
- `x_bounds`
- `y_bounds`
- `z_bounds`

Then all training STL files are projected into that shared frame to build:

- `depth_mean`
- `roughness_mean`
- `curvature_mean`
- `normal_deviation_mean`
- `density_max`
- `depth_residual_min`
- `depth_residual_max`
- `roughness_residual_max`
- `curvature_residual_max`

### 2D Projection

Each sample STL is normalized into the category coordinate frame and projected into a `256 x 256` grid. The projection stores:

- mean depth per grid cell
- vertex density per grid cell
- maximum curvature per grid cell
- mean roughness per grid cell
- mean normal vector per grid cell

Empty depth cells are filled by nearest valid neighbors so the depth map stays spatially usable.

### Per-Sample Point-Cloud Maps

The saved point-cloud descriptor maps are:

- `pc_depth`
  - signed residual from `depth_mean`
  - normalized with the category residual min/max range

- `pc_density`
  - vertex count per projected cell
  - normalized by category `density_max`

- `pc_curvature`
  - positive residual above `curvature_mean`
  - normalized by `curvature_residual_max`

- `pc_roughness`
  - positive residual above `roughness_mean`
  - normalized by `roughness_residual_max`

- `pc_normal_deviation`
  - angular deviation from the expected category normal field
  - converted to degrees
  - normalized to `[0, 1]` by dividing by `180`

### Point-Cloud Global Features

The point-cloud branch contributes:

- `pc_mean_curvature`
  - mean normalized curvature over valid projected cells

- `pc_max_roughness`
  - maximum normalized roughness over valid projected cells

- `pc_p95_normal_deviation`
  - 95th percentile of normalized normal deviation over valid projected cells

## Cross-Modal Descriptors

Cross-modal descriptors are generated in [src/preprocessing/crossmodal_descriptors.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/preprocessing/crossmodal_descriptors.py).

They do not introduce new raw sensors. Instead, they combine already-normalized IR and geometry descriptors into agreement and mismatch signals.

### Alignment Check

Before creating the final maps, the pipeline compares bounding boxes from:

- `ir_normalized > 0.02`
- `pc_depth > 0.02`

and records:

- `cross_alignment_passed`
- `cross_alignment_message`
- `cross_alignment_max_delta`

These values are written into `global.json` for auditing and debugging, but they are not part of the learned ten-dimensional global vector.

### Cross-Modal Maps

The saved cross-modal maps are:

- `cross_ir_support`
  - elementwise maximum of resized `ir_hotspot` and `ir_gradient`

- `cross_geo_support`
  - elementwise maximum of resized `pc_curvature` and `pc_roughness`

- `cross_agreement`
  - elementwise product of `cross_ir_support` and `cross_geo_support`
  - clipped to `[0, 1]`

- `cross_inconsistency`
  - absolute difference between `cross_ir_support` and `cross_geo_support`

`cross_agreement_preview.png` is only a visualization-friendly rendering of `cross_agreement`. It is not a model input.

### Cross-Modal Global Features

The cross-modal branch contributes:

- `cross_overlap_score`
  - mean value of `cross_agreement`

- `cross_centroid_distance`
  - normalized distance between top-95-percent support centroids from the IR support map and the geometry support map

- `cross_inconsistency_score`
  - mean value of `cross_inconsistency`

## Why There Are Descriptor Maps And Support Maps

MulSenDiff-X separates localized anomaly cues from support and structure cues:

- descriptor maps carry the sharper anomaly-focused signals the model can localize against
- support maps keep broader context such as normalized IR, projected depth, object occupancy, and cross-modal mismatch

This split gives the detector localized multimodal conditioning without forcing every auxiliary channel to behave like an anomaly map.

## What `global.json` Contains

`global.json` merges:

- the IR global statistics
- the point-cloud global statistics
- the cross-modal global statistics

So the file contains both:

- the ten learned global vector fields listed above
- additional audit metadata such as cross-modal alignment checks

The model only reads the ten fields defined by `GLOBAL_FEATURE_NAMES` in [src/data_loader.py](/home/eddy/Downloads/sirharginger/MulSenDiff-X-IEEE-IES-2026/MulSenDiff-X-IEEE-IES-2026/src/data_loader.py).

## Train-Only Normalization Principle

All category statistics are computed from the training split before sample folders are materialized. In practice this means:

- test samples never contribute to category normalization
- descriptor residuals are always measured against the training distribution of the same category
- app, evaluation, and training all consume the same cached descriptor representation

That design is what makes the auxiliary descriptors reproducible and defensible as conditioning inputs rather than ad hoc per-sample transforms.
