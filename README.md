# MulSenDiff-X

Diffusion-Driven Multi-Modal Industrial Anomaly Detection with detector-grounded natural-language root-cause explanation on MulSen-AD for the IEEE IES Generative AI Challenge 2026.

This repository is a **code-only reproducibility release**. It contains the implementation, scripts, and configuration templates needed to reproduce the MulSenDiff-X workflow, but it does **not** ship raw data, processed data, checkpoints, completed runs, or app-ready runtime bundles.

## What This Repo Contains

- `app/`: Streamlit demo and app-side runtime helpers
- `src/`: preprocessing, training, evaluation, retrieval, and explainability code
- `scripts/`: command-line entry points for preprocessing, training, evaluation, study orchestration, retrieval indexing, and app launch
- `config/gemini.example.json`: tracked example config for local Gemini setup
- dependency files, license, and this README

## What This Repo Does Not Contain

- `MulSen-AD` raw data
- processed sample bundles under `data/processed`
- retrieval indices under `data/retrieval`
- completed train or eval runs under `runs/`
- checkpoints or exported app bundles
- local Gemini secrets

Those assets stay local or are shared outside GitHub when needed.

## Detector Regimes

- `CCDD`: Category-Conditioned Descriptor Diffusion. One shared detector with category identity available to the conditioning path.
- `CADD`: Category-Agnostic Descriptor Diffusion. One shared detector without category identity.
- `CSDD`: Category-Specialized Descriptor Diffusion. One detector per category.

## Reproducibility Levels

### Level 1: Code Reproduction

A fresh clone is enough to:

- inspect the full implementation
- install the environment
- preprocess MulSen-AD locally
- train `CCDD`, `CADD`, and `CSDD`
- evaluate those runs locally
- build a trusted retrieval index locally

### Level 2: Full Runtime Reproduction

To run the Streamlit demo app end-to-end, a fresh clone is **not** enough. You also need local runtime assets generated or provided outside GitHub:

- processed sample bundles in `data/processed/`
- evaluation runs in `runs/`
- the checkpoints referenced by those eval runs
- optionally a retrieval index at `data/retrieval/index.jsonl`
- optionally a local Gemini config at `config/gemini.local.json`

## Dataset Acquisition

This repository does not redistribute MulSen-AD. Obtain the dataset from the original source and place it at:

```text
data/raw/MulSen_AD
```

Example:

```bash
mkdir -p data/raw
wget -O data/raw/MulSen_AD.zip "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip"
unzip data/raw/MulSen_AD.zip -d data/raw
```

If `wget` is unavailable:

```bash
mkdir -p data/raw
curl -L "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip" -o data/raw/MulSen_AD.zip
unzip data/raw/MulSen_AD.zip -d data/raw
```

## Environment Setup

### CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

### CUDA 11.8

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Verify the environment:

```bash
python -c "import sys, torch; print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
```

## Reproduce the Workflow

### 1. Preprocess MulSen-AD

```bash
python scripts/run_data_pipeline.py --processed-root data/processed
```

Expected local output:

```text
data/processed/samples/
```

### 2. Train a Detector Regime

`CCDD`:

```bash
python scripts/run_training.py --categories all --device-mode cuda --run-name main
```

`CADD`:

```bash
python scripts/run_training.py --categories all --disable-category-embedding --device-mode cuda --run-name main
```

`CSDD` example for one category:

```bash
python scripts/run_training.py --category capsule --device-mode cuda --run-name main
```

Expected local output:

```text
runs/<regime>/train/<timestamped_run>/
```

### 3. Evaluate a Trained Run

`CCDD`:

```bash
python scripts/run_evaluation.py --checkpoint runs/ccdd/train/<train_run_dir>/checkpoints/best.pt --categories all --device-mode cuda --run-name main
```

`CADD`:

```bash
python scripts/run_evaluation.py --checkpoint runs/cadd/train/<train_run_dir>/checkpoints/best.pt --categories all --disable-category-embedding --device-mode cuda --run-name main
```

`CSDD` example for one category:

```bash
python scripts/run_evaluation.py --checkpoint runs/csdd/train/<train_run_dir>/checkpoints/best.pt --category capsule --device-mode cuda --run-name main
```

Expected local output:

```text
runs/<regime>/eval/<timestamped_run>/
```

### 4. Run a Whole Regime End-to-End

```bash
python scripts/run_regime_pipeline.py --regime ccdd --device-mode cuda --run-name main
python scripts/run_regime_pipeline.py --regime cadd --device-mode cuda --run-name main
python scripts/run_regime_pipeline.py --regime csdd --device-mode cuda --run-name main
```

### 5. Run the Full Study

```bash
python scripts/run_study_pipeline.py --device-mode cuda --run-name main
```

This local workflow will preprocess if needed and then run:

- `CCDD`
- `CADD`
- `CSDD`

## Trusted Retrieval Index

The app’s RAG layer expects a locally built trusted corpus index. Create your own local reference folder and build the index into `data/retrieval/index.jsonl`.

Example:

```bash
mkdir -p docs/references
python scripts/build_trusted_corpus.py --source-root docs/references --output data/retrieval/index.jsonl
```

Nothing under `docs/` or `data/retrieval/` is shipped in this public repo by default. Those are local runtime assets.

## Streamlit App

Launch command:

```bash
python scripts/run_app.py --host 127.0.0.1 --port 8501 --headless
```

Open:

```text
http://127.0.0.1:8501
```

### Important

The app is part of the repository because it is part of the system story, but it is **not expected to run from a fresh clone alone**. It needs local runtime assets:

- processed samples in `data/processed/`
- compatible eval runs in `runs/`
- the checkpoints referenced by those eval runs
- optionally `data/retrieval/index.jsonl` for retrieval
- optionally `config/gemini.local.json` for Gemini

To configure Gemini locally:

```bash
cp config/gemini.example.json config/gemini.local.json
```

Then edit `config/gemini.local.json`:

```json
{
  "api_key": "paste_your_key_here",
  "model": "gemini-2.5-flash"
}
```

## Artifact Policy

GitHub stays code-only on purpose. Large or generated artifacts are intentionally excluded because they are:

- too large for a clean public repo
- locally reproducible from this codebase
- sometimes tied to dataset redistribution constraints
- easier to share separately when needed

If you need a runnable demo bundle for judging or private review, prepare it outside GitHub from your locally generated assets.

## Quick Checks

```bash
python scripts/run_data_pipeline.py --help
python scripts/run_training.py --help
python scripts/run_evaluation.py --help
python scripts/run_regime_pipeline.py --help
python scripts/run_study_pipeline.py --help
python scripts/build_trusted_corpus.py --help
python scripts/run_app.py --help
```

## License

See [LICENSE](LICENSE).
