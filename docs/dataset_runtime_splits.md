# MulSen-AD Dataset And Runtime Splits

## Dataset Root

MulSenDiff-X expects the extracted MulSen-AD dataset at:

- `data/raw/MulSen_AD`

This directory is treated as immutable raw data.

## Categories And Modalities

The local MulSen-AD layout used by this repo contains 15 categories:

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

Each category contains aligned:

- `RGB`
- `Infrared`
- `Pointcloud`

The alignment assumption is instance-level: the RGB image, infrared image, and point cloud for a sample refer to the same physical object instance.

## Native MulSen-AD Split Semantics

MulSen-AD itself provides:

- `train/`: normal-only samples
- `test/`: normal and anomalous samples
- `GT/`: localisation assets for evaluation only

The repo keeps the benchmark discipline:

- do not train on anomalous `test` samples
- do not train on `GT` masks
- do not use official test data for checkpoint selection

## Repo Runtime Splits

MulSenDiff-X now builds four runtime subsets per category from the native dataset:

- `train_core_good`
- `calibration_good`
- `synthetic_validation`
- `official_test`

These are **repo-generated manifests**, not extra folders inside MulSen-AD.

### How They Are Used

- `train_core_good`
  - deterministic subset of normal training samples used for model fitting

- `calibration_good`
  - held-out normal training subset used for operating calibration and false-positive control

- `synthetic_validation`
  - synthetic anomalies generated only from `calibration_good`
  - used for checkpoint and score-selection support

- `official_test`
  - the untouched MulSen-AD test split
  - used only for final evaluation

In short:

- train on `train_core_good`
- calibrate/select on `calibration_good + synthetic_validation`
- evaluate on `official_test`

## Generated Manifest Files

The runtime split builder writes manifests such as:

- `train_core_good_manifest.csv`
- `calibration_good_manifest.csv`
- `synthetic_validation_manifest.csv`
- `official_test_manifest.csv`
- `selection_manifest.csv`

These are generated under a run-local manifests directory so each training or evaluation run keeps its own split provenance.

## Synthetic Validation

The synthetic validation set is generated only from `calibration_good`, not from official test samples.

Current synthetic recipe families are:

- `thermal_hotspot`
- `geometric_surface_defect`
- `cross_modal_inconsistency`

This keeps checkpoint selection off the official test split while still providing anomaly-bearing validation examples.

## Data Management Rules

Use the following directory policy:

- `data/raw/`: untouched source data
- `data/processed/`: descriptors, projections, processed samples, derived assets
- `runs/<timestamp>_<name>/manifests/`: run-local runtime manifests

Never write generated files back into `data/raw/MulSen_AD`.

## Important Practical Note

If a document or command refers to:

- `train_core_good`
- `calibration_good`
- `synthetic_validation`
- `official_test`

that refers to the repo’s runtime manifests, not to additional directories inside the MulSen-AD download.
