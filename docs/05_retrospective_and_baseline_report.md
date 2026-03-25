# MulSenDiff-X Retrospective And Baseline Report

This document records the current state of the project, the major problems encountered during preprocessing and training, the fixes applied, the resulting category-wise baseline performance, and the main lessons that now guide the next phase of the work.

It is intended to be the "project memory" document: a place to understand what the project is trying to do, what went wrong, what was fixed, what is still weak, and why the current baseline should be treated as the working reference point.

The repo story this document supports is:

1. build and validate per-category baselines across all 15 categories
2. let those runs expose the real system problems
3. fix the problems and lock a strong reference baseline
4. scale the same MulSenDiff-X system into shared 15-category mode

## 1. Project Frame

The guiding project direction is:

- RGB-led user experience
- multimodal backend reasoning

This means:

- the operator-facing view is centered on the RGB image
- the detector remains grounded by auxiliary infrared and point-cloud descriptors
- the explanation layer is meant to explain evidence produced by the detector, not replace weak detection with language

The training protocol remains benchmark-strict:

- train on `train/good` only
- evaluate on full `test` = `test/good + test/anomaly`
- keep anomalous samples out of detector training

## 2. What The System Is Doing Now

At the current stage, the system does the following:

1. preprocesses raw RGB, infrared, and point-cloud data into aligned descriptor artifacts
2. builds processed samples and run-local train/eval manifests
3. trains a descriptor-conditioned RGB diffusion detector in either per-category or shared-category mode
4. scores anomalies with a mixture of residual, noise, and descriptor evidence
5. packages evidence for later explanation

The current implementation is still a multimodal teacher, but it now supports two operating modes inside the same code path:

- per-category baseline mode for reference validation
- shared 15-category mode for joint training and evaluation

It is not yet an RGB-only student.

## 3. Major Problems We Hit And How We Solved Them

### 3.1. Preprocessing was too slow

Observed problem:

- the data pipeline took far too long to finish
- the category-stat stage appeared to hang before sample images were written
- point-cloud processing was the main bottleneck

Root causes:

- repeated STL parsing
- repeated point-cloud feature computation during category-stat construction
- expensive geometry operations dominating the early preprocessing stage

What we changed:

- reduced repeated point-cloud work in `src/preprocessing/pointcloud_projector.py`
- improved category-stat reuse in `src/preprocessing/category_stats.py`
- added progress logging in `scripts/run_data_pipeline.py`, `src/preprocessing/category_stats.py`, and `src/preprocessing/descriptor_pipeline.py`

Effect:

- preprocessing became easier to monitor
- the point-cloud path became more tractable
- the pipeline stopped feeling "stuck in silence"

### 3.2. Sample generation failed on incomplete folders

Observed problem:

- the descriptor audit crashed with missing `meta.json`
- partially created sample folders were being audited as if they were complete

What we changed:

- hardened sample cleanup and retry behavior in `src/preprocessing/descriptor_pipeline.py`
- made the audit stage tolerant of missing files in `src/preprocessing/descriptor_audit.py`

Effect:

- incomplete sample directories stopped crashing the audit
- failed samples became recoverable instead of poisoning later stages

### 3.3. Cross-modal alignment failures blocked sample generation

Observed problem:

- some sample generations failed with cross-modal alignment tolerance errors
- this prevented sample outputs and global stats from being written

What we changed:

- changed the cross-modal path so alignment mismatches became warnings rather than fatal aborts in `src/preprocessing/crossmodal_descriptors.py`

Effect:

- descriptor PNGs and `global.json` could still be generated
- the pipeline stopped failing entirely on small alignment imperfections

### 3.4. Point-cloud projections were collapsing into tiny dark blobs

Observed problem:

- point-cloud-derived maps were almost completely dark
- `cross_geo_support` and related maps looked empty or nearly empty
- many meshes were effectively collapsing into a tiny image footprint

Root cause:

- projection used category-wide XY bounds
- large cross-sample translation differences caused per-sample meshes to shrink into very few pixels

What we changed:

- recentered point-cloud XY per sample before rasterization in `src/preprocessing/pointcloud_projector.py`

Effect:

- point-cloud maps became spatially meaningful
- geo-support and cross-modal maps gained actual structure

### 3.5. `cross_agreement.png` stayed black

Observed problem:

- after fixing the point-cloud projection, `cross_agreement.png` was still black on normal samples

Root cause:

- `cross_ir_support` relied only on `ir_hotspot`
- normal samples often had zero hotspot response
- zero IR support forced zero agreement even when geometric support existed

What we changed:

- changed cross-modal IR support to use a softer cue based on `max(ir_hotspot, ir_gradient)` in `src/preprocessing/crossmodal_descriptors.py`
- added a contrast-enhanced preview image for human inspection without changing the raw modeling artifact

Effect:

- `cross_agreement.png` stopped being uniformly black
- the raw modeling artifact remained intact
- human inspection became much easier through the preview

### 3.6. CUDA setup was blocked

Observed problem:

- CUDA appeared unavailable despite the machine having an NVIDIA GPU

Root causes:

- Secure Boot module key was not enrolled
- the first PyTorch CUDA build available in the project environment did not support the Quadro P1000 `sm_61` architecture

What we changed:

- enrolled the NVIDIA module key through MOK so the driver could load
- confirmed driver activation with `nvidia-smi`
- created a separate GPU-capable environment using a PyTorch build compatible with the Quadro P1000

Effect:

- category training moved from CPU to CUDA
- experimentation speed improved substantially

### 3.7. Reconstruction panels were misleading

Observed problem:

- the saved "reconstruction" image looked noisy and often appeared to be getting worse
- this made it hard to know whether the model was really improving

Root cause:

- the displayed reconstruction was the multi-timestep diffusion reconstruction averaged across several timesteps
- that image is useful for scoring, but not ideal as a human-facing quality panel

What we changed:

- changed visualization to show `Autoencoded RGB` as the main clean reconstruction
- kept the diffusion reconstruction as a separate comparison panel
- updated panel rendering and evaluation visualization paths

Effect:

- reconstruction quality became easier to interpret
- we stopped confusing a scoring artifact with the autoencoder quality

### 3.8. The detector collapsed after warmup

Observed problem:

- image AUROC was strong in early epochs and then fell sharply
- the model looked good early, but the later training regime degraded performance

Root cause:

- the original score schedule made an abrupt shift from descriptor-heavy warmup weights to a more aggressive residual/noise-heavy main regime

What we changed:

- replaced the hard switch with a linear interpolation schedule in `src/training/train_diffusion.py`
- later reduced the aggressiveness of the default late-stage target weights in `config/diffusion.yaml`

Effect:

- the catastrophic epoch-transition collapse was removed
- later training became more controlled, though not perfect for every category

### 3.9. Diffusion reconstruction was too soft and under-supervised

Observed problem:

- reconstructions improved numerically but still looked soft
- the diffusion branch lacked crisp detail

Root cause:

- the model was directly supervised on autoencoder reconstruction and noise prediction, but the diffusion RGB reconstruction path had weak direct pressure to become visually sharp

What we changed:

- added direct diffusion reconstruction loss in `src/models/mulsendiffx.py`
- later added Sobel-based edge reconstruction loss in `src/training/losses.py`
- logged `diffusion_reconstruction_loss` and `edge_reconstruction_loss` through the training pipeline

Effect:

- diffusion reconstruction quality improved
- anomaly maps stayed meaningful
- the system moved closer to a usable visual baseline

## 4. Current Baseline Architecture And Training Assumptions

The current baseline should be treated as the reference working system.

Key characteristics:

- latent diffusion on RGB
- descriptor-conditioned U-Net
- object-centric loader-time crop
- per-category masked RGB normalization
- multi-timestep anomaly scoring
- Gaussian-smoothed anomaly maps
- global descriptor calibration
- score scheduling across training
- direct diffusion reconstruction supervision
- edge-aware sharpening loss

This baseline is not final, but it is strong enough to support controlled comparison across categories.

## 5. Category-Wise Baseline Results

The latest run summary across category baselines was exported to:

- `runs/category_training_summary.csv`

At the time of writing, the latest available category results are:

| Category | Run | Status | Best AUROC | Best Epoch | Final AUROC | Max Pixel IoU | Final Pixel IoU | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| button_cell | `20260321_155528_button_cell` | completed | 0.977419 | 45 | 0.977419 | 0.147510 | 0.132238 | stable |
| capsule | `20260321_111358_capsule` | completed | 0.964583 | 19 | 0.937500 | 0.144732 | 0.134437 | moderate_drift |
| cotton | `20260321_163715_cotton` | completed | 0.977500 | 16 | 0.975000 | 0.043718 | 0.026572 | stable |
| cube | `20260321_124045_cube` | completed | 0.909756 | 50 | 0.909756 | 0.036466 | 0.021743 | stable |
| flat_pad | `20260321_171802_flat_pad` | completed | 0.700000 | 33 | 0.643333 | 0.059010 | 0.047324 | moderate_drift |
| light | `20260321_132909_light` | completed | 0.722222 | 50 | 0.722222 | 0.045042 | 0.040202 | stable |
| nut | `20260321_115212_nut` | completed | 0.958621 | 41 | 0.934483 | 0.058439 | 0.052642 | late_improving |
| piggy | `20260321_175942_piggy` | completed | 0.836667 | 48 | 0.836667 | 0.011538 | 0.006845 | stable |
| plastic_cylinder | `20260321_184913_plastic_cylinder` | completed | 0.825000 | 1 | 0.782143 | 0.159704 | 0.136772 | moderate_drift |
| screen | `20260321_193036_screen` | completed | 0.906250 | 49 | 0.906250 | 0.028199 | 0.002624 | stable |
| screw | `20260321_200358_screw` | completed | 0.196774 | 44 | 0.190323 | 0.033355 | 0.024487 | stable |
| solar_panel | `20260321_205256_solar_panel` | completed | 0.782051 | 13 | 0.743590 | 0.090824 | 0.074960 | moderate_drift |
| spring_pad | `20260321_213517_spring_pad` | completed | 0.987500 | 44 | 0.975000 | 0.030803 | 0.021317 | late_improving |
| toothbrush | `20260321_141610_toothbrush` | completed | 1.000000 | 1 | 1.000000 | 0.046911 | 0.027067 | stable |
| zipper | `20260321_221400_zipper` | incomplete | 0.850000 | 38 | 0.846667 | 0.073998 | 0.071941 | stable so far |

## 6. How To Read These Results

### 6.1. Strong categories

These categories currently look strong enough to trust as high-confidence baselines:

- toothbrush
- spring_pad
- cotton
- button_cell
- capsule
- nut

These are useful anchor categories for later shared-model experiments.

### 6.2. Medium categories

These categories are workable, but not yet among the strongest:

- cube
- screen
- zipper
- piggy

These are useful for testing generalization once the shared model is introduced.

### 6.3. Hard categories

These categories remain difficult under the present setup:

- plastic_cylinder
- solar_panel
- light
- flat_pad
- screw

These are especially important because they reveal where a shared model may struggle.

## 7. Interpretation Of The Current Baseline

The current baseline is strong enough to take seriously.

Why:

- preprocessing is no longer fundamentally broken
- cross-modal descriptors are now meaningful
- point-cloud maps are properly projected
- CUDA training is available
- reconstruction quality is substantially improved
- anomaly maps are object-aligned and informative on strong categories
- multiple categories now produce strong image-level AUROC

Why it is not yet "production-ready":

- category performance is uneven
- several categories still drift or plateau at modest quality
- at least one category (`screw`) is still failing badly
- pixel-level localization quality remains modest in many categories
- the shared all-category path still needs the same level of result reporting and stress-testing as the per-category baselines

The honest status is:

- strong research baseline: yes
- stable category prototype: yes
- shared-model foundation: yes
- production-ready universal detector: not yet

## 8. Main Lessons Learned

1. Geometry alignment matters more than expected.

Bad point-cloud projection can make the entire multimodal stack look broken even when the RGB branch is fine.

2. Human-facing reconstruction and scoring reconstruction are not the same thing.

The image we show for interpretation should not automatically be the same artifact used for anomaly scoring.

3. Descriptor-heavy early training was carrying the system.

The detector often looked best when descriptor-global evidence still had strong influence. This was not a weakness in itself; it was a clue about what the model trusted.

4. Abrupt regime changes are dangerous.

The score schedule can undo gains quickly if late-stage weights are too aggressive.

5. Some categories are inherently easier than others.

The project should not be judged only by the strongest categories, but the strongest categories are still valuable because they prove the core method can work.

6. Iterative repair worked.

The project improved not from one giant rewrite, but from a series of targeted fixes:

- preprocessing speed
- sample integrity
- cross-modal alignment tolerance
- point-cloud projection
- cross-agreement definition
- CUDA setup
- panel interpretation
- score scheduling
- diffusion supervision
- edge-aware sharpening

## 9. Transition To The Shared Model

The key project decision was not to build a separate joint-model codebase.

Instead, the repo now treats the shared 15-category model as **MulSenDiff-X in shared-category mode**:

- same core model
- same training entrypoint
- same evaluation entrypoint
- category-aware conditioning inside the same architecture

Why this matters:

- the repo stays easier to understand and reproduce
- the per-category baselines remain directly comparable to the shared run
- the shared model is clearly presented as the scaled version of the validated baseline system

The baselines still matter even after the shared mode exists:

1. they remain the credibility layer for the shared model story
2. they show which categories are strong anchors
3. they show which categories are still hard stress tests

Anchor categories for shared-model interpretation:

- `button_cell`, `cotton`, `capsule`, `nut`, `spring_pad`, `toothbrush`

Medium categories for broader stress testing:

- `cube`, `screen`, `zipper`, `piggy`

Hard categories to keep visible:

- `plastic_cylinder`, `solar_panel`, `light`, `flat_pad`, `screw`

## 10. Useful Reference Files

Core docs:

- `docs/01_dataset.md`
- `docs/02_architecture.md`
- `docs/03_descriptors.md`
- `docs/04_explainer.md`

Run summary:

- `runs/category_training_summary.csv`

Current training config:

- `config/diffusion.yaml`

Training and evaluation entry points:

- `scripts/run_training.py`
- `scripts/run_evaluation.py`

Category-run summary helper:

- `scripts/summarize_training_runs.py`

## 11. Final Takeaway

The project is no longer in the "does this even work?" stage.

It is now in the "which parts generalize, which parts fail, and how do we scale without losing what works?" stage.

That is a very different and much stronger place to be.
