# MulSen-AD Dataset Notes

## Dataset Root

The project uses the extracted dataset at:

- `data/raw/MulSen_AD`

This directory should be treated as immutable raw data.

## Verified Category List

The extracted dataset contains 15 industrial object categories:

- `button_cell`
- `capsule`
- `cotton`
- `cube`
- `flat_pad`
- `light`
- `nut`
- `piggy`
- `plastic_cylinder`
- `screen`
- `screw`
- `solar_panel`
- `spring_pad`
- `toothbrush`
- `zipper`

## Modalities

Each category contains three aligned sensing modalities:

- `RGB`
- `Infrared`
- `Pointcloud`

The alignment assumption is instance-level: the RGB image, infrared image, and point cloud for a given sample correspond to the same physical part instance.

## Split Semantics

Each modality follows the same anomaly-detection philosophy:

- `train/`: normal-only training data
- `test/`: normal and anomalous evaluation data
- `GT/`: ground-truth localisation assets for evaluation only

Training must remain unsupervised:

- do not train on anomalous `test/` samples
- do not train on `GT/` masks
- do not leak defect labels into normal-only model fitting

## Observed Aggregate Counts

Counts from the extracted local dataset:

| Modality | Train | Test | GT |
| --- | ---: | ---: | ---: |
| RGB | 1391 | 644 | 448 |
| Infrared | 1391 | 644 | 388 |
| Pointcloud | 1391 | 644 | 342 |

Observed file extensions:

- `png`: RGB, infrared, and some mask assets
- `stl`: point-cloud geometry samples
- `txt`: point-cloud GT annotations
- `csv`: RGB GT metadata files

## Category-Level Training Counts

Per-category training counts are currently consistent across modalities:

| Category | Train Samples per Modality |
| --- | ---: |
| button_cell | 90 |
| capsule | 64 |
| cotton | 78 |
| cube | 110 |
| flat_pad | 90 |
| light | 110 |
| nut | 118 |
| piggy | 110 |
| plastic_cylinder | 90 |
| screen | 69 |
| screw | 90 |
| solar_panel | 90 |
| spring_pad | 86 |
| toothbrush | 110 |
| zipper | 86 |

## Defect Label Diversity

Defect taxonomies vary by category. Example labels include:

- `broken`
- `broken_inside`
- `broken_outside`
- `color`
- `crack`
- `crease`
- `foreign_body`
- `hole`
- `scratch`
- `squeeze`
- `substandard`
- `open`
- `label`
- `bent`
- `detachment_inside`

This means the loader should not assume one shared global defect taxonomy for all categories.

## Data Management Rules

Use the following directory policy:

- `data/raw/`: untouched source data
- `data/processed/`: descriptors, projections, manifests, cached derived assets
- `data/splits/`: split manifests and experiment lists
- `data/cache/`: temporary runtime cache

Never write generated files back into `data/raw/MulSen_AD`.

## Recommended Manifest Schema

Each sample manifest row should contain:

- `category`
- `sample_id`
- `split`
- `defect_label`
- `is_anomalous`
- `rgb_path`
- `ir_path`
- `pointcloud_path`
- `rgb_gt_path`
- `ir_gt_path`
- `pointcloud_gt_path`
- `rgb_gt_csv_path`

The manifest should allow per-category training, per-modality debugging, and deterministic evaluation pairing.
