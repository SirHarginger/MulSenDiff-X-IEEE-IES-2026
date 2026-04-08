# Docs Overview

This folder is intentionally kept small for public review.

Core documents:

- `01_dataset.md`
  - dataset layout and runtime split policy
- `02_architecture.md`
  - high-level MulSenDiff-X system and regime overview
- `03_descriptors.md`
  - descriptor construction and model-facing inputs
- `04_explainer.md`
  - detector-to-RAG-LLM explanation flow
- `study_workflow.md`
  - practical preprocess, train, eval, export, and app workflow

Reference support:

- `references/README.md`
  - how the trusted-corpus metadata is organized in the public repo

Removed from the public-facing docs surface:

- duplicate dataset split notes
- duplicate architecture notes
- duplicate descriptor pipeline notes
- duplicate explainer flow notes

If you are reviewing the project for reproducibility, start with:

1. `README.md`
2. `docs/02_architecture.md`
3. `docs/study_workflow.md`
