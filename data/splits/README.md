# Split and Manifest Conventions

This folder stores split metadata and experiment manifests. It should not contain raw dataset files.

## Intended Contents

Recommended files for this directory:

- category-level train and validation manifests
- experiment-specific subset manifests
- ablation split definitions
- cached sample index files

## Split Philosophy

MulSen-AD already provides the core anomaly-detection split:

- `train/`: normal-only
- `test/`: normal and anomalous
- `GT/`: evaluation-only labels

This folder is for project-side bookkeeping, not for redefining the dataset semantics.

## Recommended Manifest Fields

Each manifest row should contain:

- `category`
- `sample_id`
- `modality_alignment_id`
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

## Recommended Files

Suggested first manifests:

- `train_manifest.csv`
- `test_manifest.csv`
- `per_category/<category>_train.csv`
- `per_category/<category>_test.csv`

## Rules

- never move or rename raw files inside `data/raw/MulSen_AD`
- manifests should point to raw paths rather than duplicating data
- keep split generation deterministic
- preserve one-to-one alignment across modalities
