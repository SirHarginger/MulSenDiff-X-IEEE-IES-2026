# MulSenDiff-X

MulSenDiff-X is an RGB-led, multimodal industrial anomaly detection system built on MulSen-AD. The model learns normal RGB appearance while using infrared and point-cloud descriptors as structured conditioning and evidence for anomaly scoring, localisation, and later explanation.

The repo tells one story with one system:

- we first validated **per-category baselines** across all 15 categories
- those baselines exposed the real problems in preprocessing, projection, reconstruction, scoring, and calibration
- we fixed them and established a strong reference baseline
- we then scaled the **same MulSenDiff-X system** into **shared 15-category mode**

This repo does not contain separate baseline and joint-model codebases. It contains one MulSenDiff-X implementation with two operating modes:

- `--category <name>` = per-category baseline mode
- `--categories <csv|all>` = shared joint mode

## Repo Map

- `src/` core model, training, inference, preprocessing, and explanation code
- `scripts/` command-line entrypoints for preprocessing, training, evaluation, and summaries
- `config/` runtime configuration
- `docs/` architecture notes and the detailed retrospective/baseline report
- `data/` raw, processed, and generated dataset artifacts
- `runs/` training and evaluation outputs, checkpoints, metrics, evidence, and run-local manifests

Important generated paths:

- `data/raw/` raw dataset, ignored in git
- `data/processed/` processed samples and descriptor artifacts, ignored in git
- `runs/` experiment outputs, ignored in git

`data/splits/*.csv` are generated artifacts, not the main experiment story. For reproducibility, each training and evaluation run snapshots its own manifests under `runs/<run>/manifests/`.

## Environment Setup

Supported setup paths:

- **CPU smoke path** for tests and small local checks
- **Quadro P1000 CUDA path** for the benchmark workflow used in this repo

Create a virtual environment first. If you use `fish`, replace `activate` with `activate.fish`.

### CPU Smoke Path

```bash
git clone <your-fork-or-repo-url>
cd MulSenDiff-X-IEEE-IES-2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

### Quadro P1000 CUDA Path

```bash
git clone <your-fork-or-repo-url>
cd MulSenDiff-X-IEEE-IES-2026
python3 -m venv .venv-p1000
source .venv-p1000/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

## Verify The Environment

Run this after activation:

```bash
python -c "import sys, torch; print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

For the CPU smoke path, `cuda_available: False` is expected. For the Quadro P1000 benchmark path, you should see `cuda_available: True` and `device_name: Quadro P1000`.

## Exact Workflow Commands

After activating your environment, the workflow below is the current command surface for the repo. The same entrypoints are used for both baseline mode and shared 15-category mode.

### 1. Preprocess The Data

This builds processed samples, category stats, reports, and the descriptor artifacts consumed by training. For the main benchmark workflow, run it once on the full dataset without restricting categories or excluding anomalous test data.

```bash
python scripts/run_data_pipeline.py
```

Key outputs:

- `data/processed/category_stats/<category>/`
- `data/processed/samples/<category>__<split>__<defect>__<index>/`
- `data/processed/manifests/`
- `data/processed/reports/`

### 2. Run One Baseline Category

Use this to reproduce one reference baseline, for example `capsule`.

```bash
python scripts/run_training.py --category capsule --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<capsule_run>/checkpoints/best.pt --category capsule --device-mode cuda
```

If you are on CPU, replace `--device-mode cuda` with `--device-mode cpu`.

The evaluation run writes:

- `metrics/evaluation.json` with the core metrics
- `metrics/per_category.json` and `metrics/per_category.csv`
- `predictions/eval/` with saved panels
- `evidence/` with packages and explanation artifacts

### 3. Run All 15 Baseline Categories

This is the baseline reference stage that produced the category-wise summary table in the retrospective report.

```bash
for c in button_cell capsule cotton cube flat_pad light nut piggy plastic_cylinder screen screw solar_panel spring_pad toothbrush zipper; do
  python scripts/run_training.py --category "$c" --device-mode cuda
done
```

### 4. Summarize Baseline Runs

This regenerates a compact baseline table from local run folders.

```bash
python scripts/summarize_training_runs.py --write-csv runs/category_training_summary.csv
```

The command prints a Markdown table and writes `runs/category_training_summary.csv`.

### 5. Train And Evaluate The Shared 15-Category Model

This is the headline system for the repo: the same MulSenDiff-X model running in shared-category mode.

```bash
python scripts/run_training.py --categories all --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --device-mode cuda
```

Shared evaluation now saves:

- pooled metrics in `metrics/evaluation.json`
- macro and per-category metrics in `metrics/evaluation.json`, `metrics/per_category.json`, and `metrics/per_category.csv`
- `metrics/preview_selection.json` describing the deterministic one-random-sample-per-category preview selection
- one saved preview panel per category in `predictions/eval/`

### Evaluate A Saved Checkpoint Later

```bash
python scripts/run_evaluation.py --checkpoint runs/<run>/checkpoints/best.pt --category capsule --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<run>/checkpoints/best.pt --categories all --device-mode cuda
```

## What To Inspect First In A Run Folder

Every run writes a self-contained snapshot under `runs/<timestamp>_<name>/`.

Start with these files:

- `config.json` exact run configuration, seed, runtime provenance, selected categories, and manifest snapshot paths
- `summary.json` high-level outcome and key metrics
- `metrics/evaluation.json` pooled metrics and, for shared runs, macro metrics
- `metrics/per_category.csv` per-category metrics
- `metrics/preview_selection.json` deterministic shared-preview selection metadata for shared runs
- `logs/history.csv` epoch-by-epoch training history

The full run layout is:

- `checkpoints/` model checkpoints, including `best.pt`
- `logs/` history, events, and batch-level training logs
- `metrics/` evaluation summaries, calibration files, and per-category reports
- `predictions/` saved anomaly panels
- `evidence/` saved evidence packages and explanations
- `manifests/` run-local train/eval manifests used for that exact experiment

Current publication-grade metric hierarchy:

- core: `image_auroc`, `image_auprc`, `pixel_auroc`, `aupro`
- support: `image_f1`, `pixel_f1`, `pixel_iou`, `mean_good_score`, `mean_anomalous_score`

## Benchmark Protocol And Limits

The benchmark discipline in this repo is strict:

- train on `train/good` only
- evaluate on full `test` = `test/good + test/anomaly`
- keep anomalous samples out of detector training

What the shared model is meant to do:

- generalize across the **15 known MulSen-AD categories**
- provide pooled, macro, and per-category evaluation

What it is **not** meant to claim:

- guaranteed generalization to completely unseen external categories
- an RGB-only backend at inference time

This is an **RGB-led user experience with multimodal backend reasoning**, not an RGB-only detector.

## Where The Full Story Lives

- `docs/02_architecture.md` system design and the two-mode MulSenDiff-X setup
- `docs/05_retrospective_and_baseline_report.md` the detailed story of what broke, how it was fixed, and how the 15-category baselines led into the shared model

## Quick Sanity Checks

```bash
python scripts/run_training.py --help
python scripts/run_evaluation.py --help
pytest -q tests
```

These are the fastest checks to confirm the repo is installed and the command surface works before running the full workflow.

If you are using the Quadro P1000 benchmark environment, a typical session looks like:

```bash
source .venv-p1000/bin/activate
python scripts/run_data_pipeline.py
python scripts/run_training.py --category capsule --device-mode cuda
python scripts/run_training.py --categories all --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --device-mode cuda
```
