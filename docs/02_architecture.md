# System Architecture

## Title

MulSenDiff-X: Descriptor-Conditioned Diffusion for Unsupervised Multi-Sensor Industrial Anomaly Detection and Evidence-Grounded Explanation

## High-Level Idea

MulSenDiff-X uses a three-stream representation:

- RGB for learned appearance modelling
- infrared for thermal evidence extraction
- point cloud for geometric evidence extraction

Instead of training a full joint generative model over all raw modalities, the system learns normal RGB appearance conditioned on structured thermal and geometric descriptors derived from the auxiliary modalities.

## End-to-End Pipeline

1. Load aligned RGB, infrared, and point-cloud samples for the same part instance.
2. Convert infrared and point cloud into:
   - dense spatial descriptor maps
   - compact global descriptor vectors
3. Encode:
   - RGB with an appearance encoder
   - descriptor maps with a spatial descriptor encoder
   - global vectors with a small projection MLP
4. Fuse these streams into a conditioning latent `c`.
5. Train a normal-only conditional diffusion model on RGB appearance under conditioning latent `c`.
6. At inference, score anomalies from poor reconstruction or low conditional likelihood.
7. Combine model residuals with descriptor evidence to support localisation and explanation.

The same MulSenDiff-X code path now runs in two modes:

- per-category baseline mode through `--category <name>`
- shared 15-category mode through `--categories <csv|all>`

The shared run is not a separate architecture file. It is the same MulSenDiff-X system with category-aware conditioning and joint training/evaluation.

## Why RGB Is the Generative Target

The implementation follows the refined workflow direction:

- RGB remains the primary learned appearance branch
- infrared and point cloud remain essential, but as descriptor-conditioned evidence sources

This reduces modelling complexity while still exploiting aligned multi-sensor information.

## Core Components

### 1. Dataset Index and Loaders

Responsibilities:

- discover aligned multimodal samples
- produce deterministic manifests
- pair evaluation samples with GT assets
- isolate train/test boundaries cleanly

### 2. Descriptor Extraction

Responsibilities:

- derive thermal evidence from infrared
- derive geometric evidence from point clouds
- standardise descriptor map shapes for the fusion model

### 3. Conditional Diffusion Detector

Responsibilities:

- learn the normal RGB appearance manifold
- condition generation on infrared and geometric evidence
- produce image-level anomaly scores and pixel-level residual maps

### 4. Scoring and Localisation

Responsibilities:

- combine diffusion residuals with descriptor-based evidence
- generate image-level anomaly scores
- produce localisation maps aligned to RGB space

### 5. Evidence-Grounded Explanation

Responsibilities:

- package anomaly region evidence
- attach descriptor summaries
- feed a structured evidence payload to Gemini for operator-facing explanation

## Training Principle

The detector is trained only on normal data.

Planned loss structure:

- diffusion denoising loss on RGB
- optional regularisation for descriptor consistency
- optional fusion stabilisation losses if needed later

No anomalous test samples or GT masks should influence training.

## Inference Principle

At inference time, the system should combine:

- RGB reconstruction error or denoising inconsistency
- conditional likelihood weakness under the descriptor context
- optional descriptor anomaly support score

A practical first scoring form is:

- RGB residual score
- plus descriptor-supported score
- plus optional fusion confidence adjustment

## Deployment Strategy

The development strategy in this repo is:

1. validate the system in per-category baseline mode first
2. use those results to expose and fix preprocessing, projection, reconstruction, scoring, and calibration issues
3. scale the same system into shared 15-category mode

Reasons the baseline-first path mattered:

- anomaly distributions differ strongly by object type
- descriptor semantics differ across categories
- per-category validation exposed failures that would have been hidden inside an early joint run

The current shared model is therefore best understood as **MulSenDiff-X in shared-category mode**, built on top of the validated per-category reference baselines.

## Product Deployment Note

The serious demo path in this repo is:

1. MulSenDiff-X performs anomaly detection, localisation, and calibrated severity/area estimation.
2. The detector exports a structured evidence payload.
3. Gemini turns that evidence into an operator-facing explanation.

Gemini is not the detector. It is the explanation layer. The current app build is a paired-sample demo, which means uploaded RGB images must resolve to known paired data so the multimodal backend remains honest.
