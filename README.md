# MulSenDiff-X

MulSenDiff-X is a multimodal industrial anomaly detection system built on **MulSen-AD**. It uses RGB appearance as the main reconstruction target and conditions scoring with thermal and geometric evidence derived from infrared and point-cloud data.

The repo supports three final detector regimes:

- `CCDD`: Category-Conditioned Descriptor Diffusion
- `CADD`: Category-Agnostic Descriptor Diffusion
- `CSDD`: Category-Specialized Descriptor Diffusion

It also includes:

- the final preprocessing pipeline
- train and evaluation workflows
- the locked practical localization setting
- the locked Archetype B score-rescue logic
- a Streamlit inspection app
- detector-first corrective RAG explanation support

## Final Repo Layout

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

Important runtime directories:

```text
data/
  raw/
  processed/
  splits/
  cache/
  retrieval/

docs/
  references/

model/
  ccdd/
  cadd/
  csdd/

runs/
  ccdd/
    train/
    eval/
  cadd/
    train/
    eval/
  csdd/
    train/
    eval/
  ablations/
    archetype_a/
    archetype_b/
  app_sessions/
```

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

## Final Workflow

The final workflow is:

1. preprocess into `data/processed`
2. train one regime
3. evaluate immediately after training
4. optionally export the selected eval bundle into `model/`
5. run the app from `model/` or directly from `runs/`

### Script Roles

- `scripts/run_data_pipeline.py` = preprocess only
- `scripts/run_training.py` = train only
- `scripts/run_evaluation.py` = eval only
- `scripts/run_regime_pipeline.py` = preprocess + train + eval for one regime
- `scripts/run_study_pipeline.py` = preprocess + train + eval for multiple regimes

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

### 3. Run The Full Study

```bash
python scripts/run_study_pipeline.py --device-mode cuda --run-name main
```

This preprocesses once, then runs:

- `ccdd`
- `cadd`
- `csdd`

### 4. Export App-Ready Model Bundles

Export a chosen eval run into `model/`:

```bash
python scripts/export_model_bundle.py --eval-run runs/ccdd/eval/<eval_run_dir> --force
```

The exported bundle is self-contained and includes:

- `summary.json`
- `metrics/`
- `evidence/`
- `checkpoint.pt`
- `config.json`

### 5. Run The App

```bash
python scripts/run_app.py
```

The app prefers exported bundles under `model/` and falls back to evaluation runs under `runs/`.

## RAG / LLM Support

MulSenDiff-X uses a detector-first explanation flow:

1. detector runs first
2. detector exports structured evidence
3. retrieval searches a narrow trusted corpus
4. Gemini generates a schema-constrained operator report

Trusted human-maintained source files belong in:

```text
docs/references/
```

Build the retrieval index into:

```text
data/retrieval/index.jsonl
```

Command:

```bash
python scripts/build_trusted_corpus.py
```

To enable live Gemini explanations:

```bash
export GOOGLE_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```

Gemini is explanation-only. It is not part of anomaly detection.

## Final Study Outputs

Main study outputs:

- `runs/ccdd/train/`
- `runs/ccdd/eval/`
- `runs/cadd/train/`
- `runs/cadd/eval/`
- `runs/csdd/train/`
- `runs/csdd/eval/`

Formal ablation outputs:

- `runs/ablations/archetype_a/`
- `runs/ablations/archetype_b/`

App session outputs:

- `runs/app_sessions/`

## Locked Study Decisions

The cleaned repo assumes these final decisions:

- canonical processed dataset path is `data/processed`
- main practical scorer is `legacy_raw`
- formal regimes are `CCDD`, `CADD`, and `CSDD`
- formal insight ablations are:
  - Archetype A preprocessing effect
  - Archetype B gate off vs gate on
- localization is kept as the accepted practical setting, not as a standing ablation family

## Sanity Checks

```bash
python scripts/run_data_pipeline.py --help
python scripts/run_training.py --help
python scripts/run_evaluation.py --help
python scripts/run_regime_pipeline.py --help
python scripts/run_study_pipeline.py --help
python scripts/export_model_bundle.py --help
python scripts/run_app.py --help
```
