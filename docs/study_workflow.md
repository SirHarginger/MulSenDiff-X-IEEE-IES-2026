# Final Study Workflow

This is the exact workflow for the cleaned repo.

The publication-facing workflow is the explicit preprocess -> train -> eval path. The wrapper scripts remain available as internal convenience commands.

## Canonical Assumptions

- preprocess only into `data/processed`
- write all train and eval outputs into `runs/`
- export selected app-ready bundles into `model/`
- keep only these formal ablation families:
  - Archetype A
  - Archetype B

## Script Roles

- `scripts/run_data_pipeline.py` = preprocess only
- `scripts/run_training.py` = train only
- `scripts/run_evaluation.py` = eval only
- `scripts/run_regime_pipeline.py` = optional convenience wrapper for auto-preprocess if needed + train + eval for one regime
- `scripts/run_study_pipeline.py` = optional convenience wrapper for auto-preprocess if needed + train + eval for multiple regimes

Use the lower-level scripts for the publication-facing workflow. Use the wrapper scripts only when you want the faster internal workflow.

## 1. Preprocess Once

```bash
python scripts/run_data_pipeline.py --processed-root data/processed
```

This produces the canonical processed dataset used by all three regimes.

## Train Only

```bash
python scripts/run_training.py --categories all --device-mode cuda --run-name main
python scripts/run_training.py --categories all --disable-category-embedding --device-mode cuda --run-name main
python scripts/run_training.py --category capsule --device-mode cuda --run-name main
```

## Eval Only

```bash
python scripts/run_evaluation.py --checkpoint runs/ccdd/train/<train_run_dir>/checkpoints/best.pt --categories all --device-mode cuda --run-name main
python scripts/run_evaluation.py --checkpoint runs/cadd/train/<train_run_dir>/checkpoints/best.pt --categories all --disable-category-embedding --device-mode cuda --run-name main
python scripts/run_evaluation.py --checkpoint runs/csdd/train/<train_run_dir>/checkpoints/best.pt --category capsule --device-mode cuda --run-name main
```

## Optional Internal Wrapper: Run One Final Regime

### `CCDD`

```bash
python scripts/run_regime_pipeline.py \
  --regime ccdd \
  --device-mode cuda \
  --run-name main
```

### `CADD`

```bash
python scripts/run_regime_pipeline.py \
  --regime cadd \
  --device-mode cuda \
  --run-name main
```

### `CSDD`

```bash
python scripts/run_regime_pipeline.py \
  --regime csdd \
  --device-mode cuda \
  --run-name main
```

`run_regime_pipeline.py` does:

1. preprocess only if `data/processed` is missing or incomplete
2. train
3. evaluate immediately from the new best checkpoint

Use `--overwrite-preprocess` to force a rebuild, or `--skip-preprocess` to bypass the check entirely.

Outputs:

- `runs/ccdd/train/...`
- `runs/ccdd/eval/...`
- `runs/cadd/train/...`
- `runs/cadd/eval/...`
- `runs/csdd/train/...`
- `runs/csdd/eval/...`

## Optional Internal Wrapper: Run The Full Study

```bash
python scripts/run_study_pipeline.py \
  --device-mode cuda \
  --run-name main
```

This checks `data/processed`, preprocesses only when needed, then runs:

- `ccdd`
- `cadd`
- `csdd`

You can restrict or reorder regimes if needed:

```bash
python scripts/run_study_pipeline.py --regimes ccdd,cadd
```

## 4. Export App-Ready Bundles

After choosing the eval run you want to serve in the app:

```bash
python scripts/export_model_bundle.py \
  --eval-run runs/ccdd/eval/<eval_run_dir> \
  --force
```

For `CSDD`, the export goes into a category subfolder under `model/csdd/`.

## 5. Run The App

```bash
python scripts/run_app.py
```

The app lookup order is:

1. exported bundles in `model/`
2. evaluation runs in `runs/*/eval`

## 6. Build Trusted Retrieval Support

Place approved source documents in:

```text
docs/references/
```

Then build the retrieval index:

```bash
python scripts/build_trusted_corpus.py
```

This writes:

```text
data/retrieval/index.jsonl
```

## Formal Ablations

Only keep these as formal publication ablations:

- `runs/ablations/archetype_a/`
- `runs/ablations/archetype_b/`

Localization is treated as a locked practical setting, not as a standing ablation family.
