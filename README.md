# MulSenDiff-X

MulSenDiff-X is a multimodal industrial anomaly detection system built on **MulSen-AD**. It uses RGB appearance as the main reconstruction target and conditions scoring with thermal and geometric evidence derived from infrared and point-cloud data.

The repo supports three detector regimes:

- `CCDD`: Category-Conditioned Descriptor Diffusion
- `CADD`: Category-Agnostic Descriptor Diffusion
- `CSDD`: Category-Specialized Descriptor Diffusion

It also includes:

- the preprocessing pipeline
- train and evaluation workflows
- a Streamlit inspection app

## Repo Layout

```text
app/
config/
data/
docs/
model/
runs/
scripts/
src/
```

## Public Release Contents

This public release is a **lightweight code + results + best-checkpoints repo**.

Included:

- cleaned source code, app, scripts, and docs
- lightweight final eval artifacts for:
  - `CCDD`
  - `CADD`
  - all 15 `CSDD` categories
- published best checkpoints under `model/`
- the prebuilt retrieval index at `data/retrieval/index.jsonl`
- the public Gemini config template at `config/gemini.example.json`

Intentionally omitted:

- `data/raw/`
- `data/processed/`
- full `runs/*/train/`
- epoch-by-epoch checkpoints
- bulky eval `predictions/`
- detailed `evidence/packages/` and `evidence/reports/`
- `runs/app_sessions/`
- local-only configs and temporary tooling files

Heavy private artifacts remain in local storage / external drive and are not part of the GitHub release.

### Public `runs/` policy

The published `runs/` tree contains only the final **lightweight eval artifacts**. For each kept eval run, the public repo retains:

- `summary.json`
- `metrics/`
- `plots/`
- `manifests/manifest_summary.json`
- top-level `evidence/*.json` calibration/index files used by the app and provenance flow

The public repo excludes heavy eval internals such as:

- `predictions/`
- `evidence/packages/`
- `evidence/reports/`
- `logs/`
- `checkpoints/`

### Public `model/` policy

The published `model/` tree is the checkpoint surface for the release:

- `model/ccdd/checkpoint.pt`
- `model/cadd/checkpoint.pt`
- `model/csdd/<category>/checkpoint.pt`

Each published checkpoint is accompanied by:

- `config.json`
- `summary.json`
- `bundle_manifest.json`

These model entries point back to the original source train/eval runs through their bundle manifest metadata.


## Dataset

This repository contains code only. It does **not** redistribute the MulSen-AD dataset.

Expected raw dataset location:

```text
data/raw/MulSen_AD
```

Example download flow:

```bash
mkdir -p data/raw
wget -O data/raw/MulSen_AD.zip "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip"
unzip data/raw/MulSen_AD.zip -d data/raw
```

If `wget` is unavailable:

```bash
curl -L "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip" -o data/raw/MulSen_AD.zip
unzip data/raw/MulSen_AD.zip -d data/raw
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

## Workflow

The final workflow is:

1. preprocess into `data/processed`
2. train one regime
3. evaluate immediately after training
4. optionally export or copy the selected best checkpoint into `model/`
5. run the app from `model/` or directly from `runs/`

### Script Roles

- `scripts/run_data_pipeline.py` = preprocess only
- `scripts/run_training.py` = train only
- `scripts/run_evaluation.py` = eval only
- `scripts/run_regime_pipeline.py` = auto-preprocess if needed + train + eval for one regime
- `scripts/run_study_pipeline.py` = auto-preprocess if needed + train + eval for multiple regimes

So if you want full automation, use the pipeline scripts. If you want tighter control, you can still run preprocess, train, and eval separately.

### 1. Preprocess

```bash
python scripts/run_data_pipeline.py --processed-root data/processed
```

### Train Only

Examples:

```bash
python scripts/run_training.py --categories all --device-mode cuda --run-name main
python scripts/run_training.py --categories all --disable-category-embedding --device-mode cuda --run-name main
python scripts/run_training.py --category capsule --device-mode cuda --run-name main
```

These write training runs under:

- `runs/ccdd/train/`
- `runs/cadd/train/`
- `runs/csdd/train/`

### Eval Only

Examples:

```bash
python scripts/run_evaluation.py --checkpoint runs/ccdd/train/<train_run_dir>/checkpoints/best.pt --categories all --device-mode cuda --run-name main
python scripts/run_evaluation.py --checkpoint runs/cadd/train/<train_run_dir>/checkpoints/best.pt --categories all --disable-category-embedding --device-mode cuda --run-name main
python scripts/run_evaluation.py --checkpoint runs/csdd/train/<train_run_dir>/checkpoints/best.pt --category capsule --device-mode cuda --run-name main
```

These write evaluation runs under:

- `runs/ccdd/eval/`
- `runs/cadd/eval/`
- `runs/csdd/eval/`

### 2. Run One Regime

```bash
python scripts/run_regime_pipeline.py --regime ccdd --device-mode cuda --run-name main
python scripts/run_regime_pipeline.py --regime cadd --device-mode cuda --run-name main
python scripts/run_regime_pipeline.py --regime csdd --device-mode cuda --run-name main
```

What each regime does:

- `ccdd`: one shared category-conditioned model
- `cadd`: one shared model without category identity
- `csdd`: one per-category model per class, each followed by its own eval run

Outputs:

- training runs go to `runs/<regime>/train/`
- evaluation runs go to `runs/<regime>/eval/`

Preprocessing behavior:

- if `data/processed` already looks complete, the wrapper skips preprocessing automatically
- use `--overwrite-preprocess` to force a rebuild
- use `--skip-preprocess` to bypass the check entirely

### 3. Run The Full Study

```bash
python scripts/run_study_pipeline.py --device-mode cuda --run-name main
```

This checks `data/processed`, preprocesses only when needed, then runs:

- `ccdd`
- `cadd`
- `csdd`

### 4. Export Or Publish Best Checkpoints

You can export a chosen eval run into `model/`:

```bash
python scripts/export_model_bundle.py --eval-run runs/ccdd/eval/<eval_run_dir> --force
```

The export command creates a self-contained bundle with:

- `summary.json`
- `metrics/`
- `evidence/`
- `checkpoint.pt`
- `config.json`

For this public release, the published `model/` directory is lighter than a full exported bundle and keeps:

- `checkpoint.pt`
- `config.json`
- `summary.json`
- `bundle_manifest.json`

The original full train runs remain private.

## App Quick Start

MulSenDiff-X uses a detector-first explanation flow:

1. detector runs first
2. detector exports structured evidence
3. retrieval searches a narrow trusted corpus
4. Gemini generates a schema-constrained operator report


Build the retrieval index into:

```text
data/retrieval/index.jsonl
```

Rebuild the retrieval index only after adding your own approved source files:

```bash
python scripts/build_trusted_corpus.py
```

Create the local Gemini config:

```bash
cp config/gemini.example.json config/gemini.local.json
```

Then open `config/gemini.local.json` and set:

```json
{
  "api_key": "your_key_here",
  "model": "gemini-2.5-flash"
}
```


The app will read this JSON file automatically, so the terminal only needs the run command below.

Launch the app from the activated project environment:

```bash
python scripts/run_app.py --host 127.0.0.1 --port 8501 --headless
```

Then open example:

```text
http://127.0.0.1:8501
```

## Checks

```bash
python scripts/run_data_pipeline.py --help
python scripts/run_training.py --help
python scripts/run_evaluation.py --help
python scripts/run_regime_pipeline.py --help
python scripts/run_study_pipeline.py --help
python scripts/export_model_bundle.py --help
python scripts/run_app.py --help
```
