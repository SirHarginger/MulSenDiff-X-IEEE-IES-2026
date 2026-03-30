# Publication Evaluation Workflow

## Paper Story

Freeze the paper around three detector protocols:

- `Shared`
  - headline method

- `PerCat`
  - benchmark-compatible reference

- `Shared-NoCat`
  - key ablation

Keep only two main-paper extras:

- modality-visibility analysis
- descriptor robustness

Keep these out of the main paper unless explicitly needed in the appendix:

- ROC/PR curves
- score-distribution plots
- reconstruction quality
- explanation evaluation without a real human study

## Main Metrics

Paper-facing naming:

- `image_auroc` -> `object_auroc`
- `image_auprc` -> `object_auprc`

Main table metrics:

- pooled/global `object_auroc`
- macro `object_auroc`
- std across categories for `object_auroc`
- worst-category `object_auroc`
- worst-5-category mean `object_auroc`
- macro `pixel_auroc`
- macro `pixel_aupr`
- macro `pixel_f1_max`
- macro `aupro`

These are now written by the standard evaluation path and consumed by the publication exporter.

## Required Inputs

The publication exporter expects explicit run inputs, not generic summaries.

### Per-category manifest

File:

- `docs/publication/percat_eval_manifest.csv`

Columns:

- `category`
- `eval_run_dir`

### Qualitative example manifest

File:

- `docs/publication/qualitative_examples.csv`

Columns:

- `figure_id`
- `package_path`
- `thermal_map_name`
- `auxiliary_map_name`
- `title`
- `caption`

Use this to select the sample for the main multimodal qualitative figure.

### Modality-visibility mapping

File:

- `docs/publication/modality_visibility_mapping.csv`

Columns:

- `category`
- `defect_label`
- `visibility_bucket`
- `rationale`
- `review_status`

Allowed buckets:

- `rgb_visible`
- `ir_dominant_internal`
- `geometry_dominant`
- `cross_modal_overlap`
- `uncertain`

This mapping is manual by design. Do not auto-infer it from labels or scores.

### Robustness manifest

File:

- `docs/publication/robustness_manifest.csv`

Columns:

- `label`
- `eval_run_dir`

Use it to aggregate robustness eval runs into the paper-facing robustness table.

## Standard Run Order

### 1. Train and evaluate the shared model

```bash
python scripts/run_training.py --categories all --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --device-mode cuda
```

### 2. Train and evaluate the Shared-NoCat ablation

```bash
python scripts/run_training.py --categories all --disable-category-embedding --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_nocat_run>/checkpoints/best.pt --categories all --device-mode cuda
```

To confirm a run is a true `Shared-NoCat` ablation, check the saved run metadata for:

- `disable_category_embedding: true`
- `ablation_mode: "shared_nocat"`

### 3. Train and evaluate the per-category reference runs

Run one category at a time, then record each eval directory in `docs/publication/percat_eval_manifest.csv`.

Example:

```bash
python scripts/run_training.py --category capsule --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<capsule_run>/checkpoints/best.pt --category capsule --device-mode cuda
```

## Descriptor Robustness

Robustness is evaluation-only on the final shared checkpoint.

Useful options:

```bash
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation drop_ir --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation drop_pointcloud --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation drop_crossmodal --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation missing_ir --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation missing_pointcloud --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --descriptor-noise-sigma 0.10 --device-mode cuda
```

Record the resulting eval run directories in `docs/publication/robustness_manifest.csv`.

## Export The Paper Package

Use the public exporter:

```bash
python scripts/run_publication_export.py \
  --shared-eval-run runs/<shared_eval_run> \
  --shared-nocat-eval-run runs/<shared_nocat_eval_run> \
  --percat-eval-manifest docs/publication/percat_eval_manifest.csv \
  --qualitative-manifest docs/publication/qualitative_examples.csv \
  --out-dir publication
```

Optional extras:

```bash
python scripts/run_publication_export.py \
  --shared-eval-run runs/<shared_eval_run> \
  --shared-nocat-eval-run runs/<shared_nocat_eval_run> \
  --percat-eval-manifest docs/publication/percat_eval_manifest.csv \
  --qualitative-manifest docs/publication/qualitative_examples.csv \
  --visibility-mapping docs/publication/modality_visibility_mapping.csv \
  --robustness-manifest docs/publication/robustness_manifest.csv \
  --out-dir publication
```

## Expected Outputs

Core package:

- `publication/tables/table1_main_comparison.csv`
- `publication/tables/table2_per_category_shared.csv`
- `publication/figures/figure1_multimodal_case.png`
- `publication/figures/figure2_category_stability.png`
- `publication/publication_tables.xlsx`

Extended package when visibility and robustness are supplied:

- `publication/tables/table3_modality_visibility.csv`
- `publication/tables/table4_descriptor_robustness.csv`

Appendix exports when source plots are available:

- `publication/appendix/roc_curve.png`
- `publication/appendix/pr_curve.png`
- `publication/appendix/score_distribution.png`

## What Not To Do

- do not use `summary.json` alone to write paper tables
- do not auto-infer modality buckets
- do not treat explanation as part of the detector benchmark without a real human study
- do not make ROC/PR curves the main-paper figures
