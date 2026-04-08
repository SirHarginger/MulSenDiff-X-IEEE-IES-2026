# MulSenDiff-X

Code-only reproducibility release for MulSenDiff-X on MulSen-AD.

## Includes

- `app/` Streamlit demo
- `src/` preprocessing, training, evaluation, retrieval, explanation
- `scripts/` CLI entry points
- `config/gemini.example.json` example Gemini config

## Does not include

- MulSen-AD raw data
- processed data
- retrieval index
- completed runs
- checkpoints
- local secrets

## Dataset

Place MulSen-AD here:

```text
data/raw/MulSen_AD
```

Download data:

```bash
mkdir -p data/raw
wget -O data/raw/MulSen_AD.zip "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip"
unzip data/raw/MulSen_AD.zip -d data/raw
```

## Setup

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

## Regimes

* `ccdd` shared detector with category conditioning
* `cadd` shared detector without category conditioning
* `csdd` category-specific detector

## Split policy

* train on `train_core_good`
* calibrate/select on `calibration_good + synthetic_validation`
* evaluate on `official_test`

Do not use the official test split for checkpoint selection.

## Run

### Preprocess

```bash
python scripts/run_data_pipeline.py --processed-root data/processed
```

### Train

```bash
python scripts/run_training.py --categories all --device-mode cuda --run-name main
python scripts/run_training.py --categories all --disable-category-embedding --device-mode cuda --run-name main
python scripts/run_training.py --category capsule --device-mode cuda --run-name main
```

### Evaluate

```bash
python scripts/run_evaluation.py --checkpoint runs/ccdd/train/<train_run_dir>/checkpoints/best.pt --categories all --device-mode cuda --run-name main
python scripts/run_evaluation.py --checkpoint runs/cadd/train/<train_run_dir>/checkpoints/best.pt --categories all --disable-category-embedding --device-mode cuda --run-name main
python scripts/run_evaluation.py --checkpoint runs/csdd/train/<train_run_dir>/checkpoints/best.pt --category capsule --device-mode cuda --run-name main
```

### Pipelines

```bash
python scripts/run_regime_pipeline.py --regime ccdd --device-mode cuda --run-name main
python scripts/run_regime_pipeline.py --regime cadd --device-mode cuda --run-name main
python scripts/run_regime_pipeline.py --regime csdd --device-mode cuda --run-name main
python scripts/run_study_pipeline.py --device-mode cuda --run-name main
```

## Retrieval

```bash
mkdir -p docs/references
python scripts/build_trusted_corpus.py --source-root docs/references --output data/retrieval/index.jsonl
```

## App

```bash
python scripts/run_app.py --host 127.0.0.1 --port 8501 --headless
```

Open `http://127.0.0.1:8501`

The app needs local runtime assets and is not expected to run from a fresh clone alone.

## Gemini

```bash
cp config/gemini.example.json config/gemini.local.json
```

## Rules

* `data/raw/` untouched source data
* `data/processed/` generated assets
* `runs/` train and eval outputs
* do not write generated files into `data/raw/MulSen_AD`

