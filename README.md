# MulSenDiff-X

MulSenDiff-X is an RGB-led multimodal industrial anomaly detection project built on **MulSen-AD**. It models normal RGB appearance while using infrared and point-cloud descriptors as structured conditioning and evidence for anomaly scoring, localisation, and explanation.

The repository contains **one system** with two modes:

- `--category <name>`: per-category baseline mode
- `--categories <csv|all>`: shared multi-category mode

## What This Repo Contains

- training and evaluation code for MulSenDiff-X
- preprocessing and descriptor generation pipelines
- shared-model and per-category benchmark workflows
- a Streamlit inspection app with Gemini-based explanation

Generated assets are kept out of git:

- `data/raw/`
- `data/processed/`
- `runs/`

## Dataset Access

This repository contains **code only**. It does **not** include or redistribute the MulSen-AD dataset.

Dataset:

- official name: `MulSen-AD — Multi-Sensor Object Anomaly Detection Dataset`
- paper: `Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties`
- GitHub: `https://github.com/ZZZBBBZZZ/MulSen-AD`
- Hugging Face: `https://huggingface.co/datasets/orgjy314159/MulSen_AD`
- file tree: `https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main`
- archive: `MulSen_AD.zip` (`8.97 GB`)

Download and extract it into the layout expected by the code:

```bash
mkdir -p data/raw
wget -O data/raw/MulSen_AD.zip "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip"
unzip data/raw/MulSen_AD.zip -d data/raw
ls data/raw/MulSen_AD
```

If `wget` is unavailable:

```bash
curl -L "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip" -o data/raw/MulSen_AD.zip
```

## Setup

Create a virtual environment, install the repo requirements, then install the PyTorch build that matches the machine.

### CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

### CUDA

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Verify the runtime:

```bash
python -c "import sys, torch; print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## Quickstart

### 1. Preprocess the dataset

```bash
python scripts/run_data_pipeline.py
```

### 2. Train one baseline category

```bash
python scripts/run_training.py --category capsule --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<capsule_run>/checkpoints/best.pt --category capsule --device-mode cuda
```

Use `--device-mode cpu` on CPU-only machines.

### 3. Train and evaluate the shared model

```bash
python scripts/run_training.py --categories all --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --device-mode cuda
```

### 4. Summarize baseline runs

```bash
python scripts/summarize_training_runs.py --write-csv runs/category_training_summary.csv
```

### 5. Run the app

The current app is a **paired-sample inspection demo**. It is RGB-led, but the backend still resolves uploads to known paired samples so the multimodal detector can use its calibrated evidence path.

To enable live Gemini explanations:

1. Go to `https://aistudio.google.com/app/apikey`
2. Sign in with your Google account
3. Click `Create API key`
4. Set the key in your terminal

```bash
export GOOGLE_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```

Keep the key private and never commit it to git.

Then run:

```bash
python scripts/run_app.py
```

## Outputs

Each run writes a self-contained folder under `runs/<timestamp>_<name>/`.

Most important files:

- `config.json`: exact configuration and runtime provenance
- `summary.json`: headline results
- `metrics/evaluation.json`: pooled metrics and shared-run macro metrics
- `metrics/per_category.csv`: per-category breakdown
- `plots/`: training or evaluation figures
- `evidence/`: packages, calibration artifacts, and optional debug explanation artifacts

## Benchmark Protocol

- train on `train/good` only
- evaluate on full `test` = `test/good + test/anomaly`
- report core metrics: `image_auroc`, `image_auprc`, `pixel_auroc`, `aupro`

The shared model is designed to generalize across the **15 known MulSen-AD categories**. It should not be described as an RGB-only detector or as guaranteed to generalize to unseen external categories.

## Documentation

- `docs/02_architecture.md`: system design
- `docs/04_explainer.md`: explanation strategy
- `docs/05_retrospective_and_baseline_report.md`: development history, struggles, and baseline results

## Sanity Checks

```bash
python scripts/run_training.py --help
python scripts/run_evaluation.py --help
pytest -q tests
```
