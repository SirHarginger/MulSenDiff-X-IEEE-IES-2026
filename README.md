# MulSenDiff-X

MulSenDiff-X is a category-aware multimodal industrial anomaly detection project built on **MulSen-AD**. It learns normal RGB appearance while using sensor-derived thermal-geometric descriptors from infrared and point-cloud data for conditioning, scoring, localisation, and evidence-grounded explanation.

The repo supports three detector workflows:

- `PerCat`: one category-specific model
- `Shared`: one shared model across all 15 categories
- `Shared-NoCat`: shared training without category identity, used as the key ablation

## What This Repo Contains

- data preprocessing and descriptor generation
- training and evaluation code for MulSenDiff-X
- shared-model and per-category benchmark workflows
- a Streamlit inspection app with Gemini-based explanation

Generated assets are kept out of git:

- `data/raw/`
- `data/processed/`
- `runs/`
- `publication/`

## Get The Code

```bash
git clone <repository-url>
cd MulSenDiff-X-IEEE-IES-2026
```

Run all commands below from the repository root.

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

Use the same activated environment for the remaining commands.

## Quickstart

### 1. Preprocess the dataset

```bash
python scripts/run_data_pipeline.py
```

### 2. Train and evaluate one per-category reference model

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

### 4. Train and evaluate the Shared-NoCat ablation

```bash
python scripts/run_training.py --categories all --disable-category-embedding --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_nocat_run>/checkpoints/best.pt --categories all --device-mode cuda
```

### 5. Run evaluation-only robustness ablations

Examples:

```bash
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation drop_ir --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --conditioning-ablation drop_pointcloud --device-mode cuda
python scripts/run_evaluation.py --checkpoint runs/<shared_run>/checkpoints/best.pt --categories all --descriptor-noise-sigma 0.10 --device-mode cuda
```

### 6. Run the app

The current app is a **paired-sample inspection demo**. It is RGB-led, but the backend resolves uploads to known paired samples so the detector can use its multimodal evidence path honestly.

Gemini is used only for explanation. The detector remains the source of truth.

To enable live Gemini explanations:

1. Go to `https://aistudio.google.com/app/apikey`
2. Sign in with your Google account
3. Click `Create API key`
4. Set the key in your terminal

```bash
export GOOGLE_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```

Then run:

```bash
python scripts/run_app.py
```

## Training And Evaluation Protocol

The corrected repo workflow is:

- train on `train_core_good`
- calibrate and select checkpoints on `calibration_good + synthetic_validation`
- evaluate on `official_test`

## Publication Export

For paper-ready tables and figures, use:

```bash
python scripts/run_publication_export.py --help
```


## Outputs

Each run writes a self-contained folder under `runs/<timestamp>_<name>/`.

Most important files:

- `config.json`: exact configuration and runtime provenance
- `summary.json`: headline results
- `metrics/evaluation.json`: pooled and macro metrics
- `metrics/per_category.csv`: per-category breakdown
- `plots/`: training or evaluation figures
- `evidence/`: evidence packages and explanation artifacts

Publication exports are written separately under `publication/`.

## Sanity Checks

```bash
python scripts/run_training.py --help
python scripts/run_evaluation.py --help
python scripts/run_publication_export.py --help
pytest -q tests
```
