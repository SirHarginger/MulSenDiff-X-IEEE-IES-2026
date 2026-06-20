# MulSenDiff-X

Descriptor-conditioned diffusion for unsupervised multi-sensor industrial anomaly
detection on MulSen-AD. The code supports RGB, infrared, and point-cloud
descriptors, with shared and category-specialized detector regimes plus
sensor-traceable evaluation artifacts.

This is a code-only reproducibility release. It does not include MulSen-AD raw
data, processed descriptors, trained checkpoints, completed runs, retrieval
indexes, or local API secrets.

## Repository Layout

- `src/`: preprocessing, training, inference, evaluation, retrieval, explanation
- `scripts/`: command-line entry points for data, training, evaluation, studies
- `app/`: optional Streamlit demo
- `config/`: diffusion and local-service configuration templates
- `runs/`: generated training/evaluation outputs, ignored by release policy

## Setup

Use Python 3.12. CUDA users should verify `nvidia-smi` and
`torch.cuda.is_available()` before launching long runs.

```bash
python3.12 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

CPU-only install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

## Data

Place MulSen-AD at:

```text
data/raw/MulSen_AD
```

Example download layout:

```bash
mkdir -p data/raw
wget -O data/raw/MulSen_AD.zip "https://huggingface.co/datasets/orgjy314159/MulSen_AD/resolve/main/MulSen_AD.zip"
unzip data/raw/MulSen_AD.zip -d data/raw
```

Build processed descriptors:

```bash
python scripts/run_data_pipeline.py --processed-root data/processed
```

## Regimes

- `ccdd`: shared descriptor diffusion with category conditioning
- `cadd`: shared descriptor diffusion without category conditioning
- `csdd`: category-specialized descriptor diffusion

Split policy: train on `train_core_good`, calibrate/select on
`calibration_good + synthetic_validation`, and report on `official_test`.

## Core Runs

Train/evaluate one regime:

```bash
python scripts/run_regime_pipeline.py --regime ccdd --skip-preprocess --processed-root data/processed --device-mode cuda --run-name main --max-visualizations 0
python scripts/run_regime_pipeline.py --regime cadd --skip-preprocess --processed-root data/processed --device-mode cuda --run-name main --max-visualizations 0
python scripts/run_regime_pipeline.py --regime csdd --skip-preprocess --processed-root data/processed --device-mode cuda --run-name main --max-visualizations 0
```

Run the full three-regime study:

```bash
python scripts/run_study_pipeline.py --regimes ccdd,cadd,csdd --skip-preprocess --processed-root data/processed --device-mode cuda --seed 17 --run-name camera_ready_seed17 --max-visualizations 0
python scripts/run_study_pipeline.py --regimes ccdd,cadd,csdd --skip-preprocess --processed-root data/processed --device-mode cuda --seed 37 --run-name camera_ready_seed37 --max-visualizations 0
```

Training is storage-light by default: each run writes `checkpoints/best.pt` for
evaluation and `checkpoints/last.pt` for resuming. It does not keep every
`epoch_XXX.pt` checkpoint unless `--save-epoch-checkpoints` is passed.

Resume a run:

```bash
python scripts/run_regime_pipeline.py --regime csdd --categories cube --skip-preprocess --processed-root data/processed --device-mode cuda --seed 17 --run-name camera_ready_seed17_resume --max-visualizations 0 --resume-checkpoint runs/csdd/train/<partial_run>/checkpoints/last.pt
```

If the interrupted run predates `last.pt`, resume from `best.pt` or the newest
available `epoch_XXX.pt`.

## Evaluation

Evaluate a saved checkpoint directly:

```bash
python scripts/run_evaluation.py --checkpoint runs/ccdd/train/<run>/checkpoints/best.pt --categories all --processed-root data/processed --device-mode cuda --run-name eval
python scripts/run_evaluation.py --checkpoint runs/cadd/train/<run>/checkpoints/best.pt --categories all --disable-category-embedding --processed-root data/processed --device-mode cuda --run-name eval
python scripts/run_evaluation.py --checkpoint runs/csdd/train/<run>/checkpoints/best.pt --category capsule --processed-root data/processed --device-mode cuda --run-name eval
```

Required eval artifacts are `summary.json`, `metrics/evaluation.json`, and
`metrics/per_category.json`.

## Camera-Ready Aggregation

Create a manifest with explicit seed-to-run paths, then aggregate:

```bash
python scripts/summarize_camera_ready.py --manifest runs/camera_ready_manifest.json --output-root runs/camera_ready_summary --copy-manifest
```

Outputs include:

`main_multiseed_summary.csv`, `tables/main_multiseed_table.tex`,
`ccdd_cadd_paired_delta_summary.csv`, `tables/mechanism_ablation_table.tex`,
split figure assets, and `source_runs.csv`.

## Mechanism Ablations

Thermal/internal-defect rescue gate ablation. Run CCDD gate-on evaluation for
the affected categories only, once per seed/checkpoint:

```bash
python scripts/run_evaluation.py --checkpoint <ccdd_seed_best.pt> --categories capsule,solar_panel --processed-root data/processed --device-mode cuda --seed <seed> --run-name gate_on_seed<seed> --enable-internal-defect-gate --max-visualizations 0
```

Reference-statistic substitution ablation. Copy processed descriptors, rebuild
only `screw,zipper` with the archetype-A reference policy disabled, then train
CCDD on the copied processed root:

```bash
cp -a data/processed data/processed_no_refsub
python scripts/run_data_pipeline.py --processed-root data/processed_no_refsub --categories screw,zipper --overwrite --skip-manifests --disable-archetype-a-reference-policy
grep -E 'screw|zipper' data/processed_no_refsub/reports/descriptor_policy_audit.csv
python scripts/run_regime_pipeline.py --regime ccdd --skip-preprocess --processed-root data/processed_no_refsub --device-mode cuda --seed <seed> --run-name refsub_off_seed<seed> --max-visualizations 0
```

Add ablation eval paths to `runs/camera_ready_manifest.json` under
`ablations.gate_on.eval_runs` or `ablations.refsub_off.eval_runs`, then rerun
`scripts/summarize_camera_ready.py`.

## Additional Studies

Held-out category generalization:

```bash
python scripts/run_generalization_study.py --phase full --closed-set-eval-run runs/ccdd/eval/<closed_set_eval_run> --device-mode cuda --run-name revision
```

Explanation audit:

```bash
python scripts/run_explanation_ablation.py --phase full --eval-run runs/ccdd/eval/<closed_set_eval_run> --knowledge-base-root data/retrieval --run-name revision
```

The explanation workflow writes audit manifests, blinded rating templates, mode
outputs, and qualitative comparison panels under `runs/ccdd/explanation_ablation/`.

## Retrieval And App

Build a local trusted corpus index:

```bash
mkdir -p docs/references
python scripts/build_trusted_corpus.py --source-root docs/references --output data/retrieval/index.jsonl
```

Run the demo:

```bash
python scripts/run_app.py --host 127.0.0.1 --port 8501 --headless
```

Open `http://127.0.0.1:8501`. The app needs local data/checkpoint assets and
is not expected to run from a fresh clone alone.

## Local Secrets

For Gemini-backed explanation experiments:

```bash
cp config/gemini.example.json config/gemini.local.json
```

Keep `config/gemini.local.json` and other secrets out of version control.
