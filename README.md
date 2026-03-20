# MulSenDiff-X: Descriptor-Conditioned Diffusion for Unsupervised Multi-Sensor Industrial Anomaly Detection and Evidence-Grounded Explanation

MulSenDiff-X is the implementation repo for an IEEE IES Generative AI Challenge project built on the MulSen-AD dataset. The project uses RGB as the learned appearance branch and turns infrared thermography plus 3D point clouds into descriptor-based conditioning signals for anomaly detection, localisation, and explanation.

## Project Direction

The core idea is:

- learn normal RGB appearance with a conditional diffusion model
- derive thermal and geometric evidence from aligned infrared and point-cloud data
- use those descriptors to condition detection and support localisation
- package the resulting evidence for a grounded explanation module

This keeps the detector fully unsupervised during training while still using all three aligned sensing modalities at inference and evaluation time.

## Dataset Status

The dataset is present locally at:

- `data/raw/MulSen_AD`

Verified local structure:

- 15 object categories
- 3 aligned modalities per category: `RGB`, `Infrared`, `Pointcloud`
- normal-only training splits under `train/`
- evaluation data under `test/`
- modality-specific ground truth under `GT/`

Observed aggregate counts from the extracted dataset:

- RGB: `1391` train files, `644` test files, `448` GT files
- Infrared: `1391` train files, `644` test files, `388` GT files
- Pointcloud: `1391` train files, `644` test files, `342` GT files
- extra metadata assets: `87` CSV files in RGB GT folders

## Current Repo Status

The repo is no longer just scaffolding. It now contains:

- dataset indexing and descriptor preprocessing
- a trainable descriptor-conditioned diffusion baseline
- evaluation and anomaly scoring
- evidence packaging for explanation
- a checkpoint-driven Streamlit interface

## Repo Layout

- `src/`: core pipeline code
- `app/`: user-facing interface
- `scripts/`: CLI entrypoints for preprocessing, training, and evaluation
- `config/`: live runtime config
- `docs/`: project and design notes
- `tests/`: runtime verification

Generated artifacts live outside the source tree:

- `data/raw/`
- `data/processed/`
- `data/cache/`
- `data/splits/*.csv`
- `runs/`

These paths are ignored in `.gitignore` so the repository stays focused on source code and documentation.

## Main Entry Points

Pipeline:

- `scripts/run_data_pipeline.py`
- `scripts/run_training.py`
- `scripts/run_evaluation.py`
- `scripts/run_app.py`

Interface:

- `python scripts/run_app.py`

## What To Run Chronologically

From the repo root with the project venv active, run the project in this order.

### 1. Build data artifacts

This step builds the dataset index, descriptor pipeline outputs, validation reports, audits, and model manifests.

```bash
python scripts/run_data_pipeline.py
```

### 2. Train one category

Start with one category such as `capsule` while iterating on the model.

```bash
python scripts/run_training.py --category capsule --epochs 20 --batch-size 4 --image-size 256 --device cpu --run-name capsule_clean
```

### 3. Evaluate the trained checkpoint

Use the `best.pt` checkpoint from the training run you just created.

```bash
python scripts/run_evaluation.py --checkpoint runs/<run>/checkpoints/best.pt --category capsule --batch-size 4 --image-size 256 --device cpu --run-name capsule_eval
```

### 4. Launch the app

The app loads trained checkpoints and runs live inference from the model.

```bash
python scripts/run_app.py
```

### 5. Scale up after the first category works

When the pipeline looks good on `capsule`, repeat training and evaluation for other categories.

## Hardware Note

CUDA is not currently available on this machine because Secure Boot key enrollment was not completed after driver installation. The Ubuntu driver stack is now installed cleanly with `nvidia-driver-535`, but the MOK approval step still needs to be accepted during reboot before NVIDIA modules can load.

Until that is completed, assume:

- CPU-first development
- small-scale preprocessing and indexing locally
- conservative training settings
- optional later migration to a stronger GPU machine for full experiments

## Keep In Mind

- `data/` and `runs/` are working directories, not source code
- the main workflow is `run_data_pipeline -> run_training -> run_evaluation -> run_app`
- the app is checkpoint-driven and runs inference from trained models
- the detailed design notes live in `docs/01_dataset.md` through `docs/04_explainer.md`
