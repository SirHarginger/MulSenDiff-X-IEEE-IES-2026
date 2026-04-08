# Published Evaluation Runs

This public release keeps only the final **lightweight evaluation artifacts** for:

- `CCDD`
- `CADD`
- all 15 `CSDD` category-specific evaluations

Each published eval run is intentionally trimmed to reviewer-facing artifacts:

- `summary.json`
- `metrics/`
- `plots/`
- `manifests/manifest_summary.json`
- top-level `evidence/*.json` calibration and index files

Heavy run artifacts are intentionally excluded from Git tracking:

- `predictions/`
- `evidence/packages/`
- `evidence/reports/`
- `logs/`
- `checkpoints/`
- all `train/` runs
- `app_sessions/`

The omitted heavy assets remain local/private and are not part of the public GitHub release.
