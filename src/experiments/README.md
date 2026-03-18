# Experiment Roadmap

This directory should hold experiment notes, ablation definitions, and lightweight configuration records for MulSenDiff-X.

## Phase 1: Data and Preprocessing

Goals:

- verify sample alignment across modalities
- build train/test manifests
- validate descriptor extraction outputs

Deliverables:

- manifest generator
- loader smoke tests
- descriptor cache prototype

## Phase 2: Baseline Detector

Goals:

- train a per-category conditional RGB diffusion baseline
- use infrared and point cloud as descriptor conditioning only

Deliverables:

- first trainable detector
- image-level anomaly scores
- basic residual-based localisation maps

## Phase 3: Descriptor Ablations

Goals:

- test RGB-only baseline
- test RGB plus infrared descriptors
- test RGB plus point-cloud descriptors
- test full descriptor-conditioned model

Deliverables:

- ablation table
- descriptor contribution analysis

## Phase 4: Explanation Pipeline

Goals:

- package evidence from detector outputs
- produce templated grounded explanations
- prepare optional retrieval and LLM integration

Deliverables:

- evidence bundle schema
- explanation examples
- evaluation notes for grounded explanation quality

## Phase 5: Challenge-Ready Reporting

Goals:

- consolidate per-category metrics
- curate qualitative examples
- document limitations and compute constraints

Deliverables:

- benchmark summary
- visualisations
- report-ready outputs
