# Literature Review Notes

## Purpose of This File

This file is an implementation-oriented literature frame for the project, not a final paper-ready bibliography. It captures the research directions that the codebase should align with. Exact citations should be verified and inserted before manuscript submission.

## Research Threads Relevant to MulSenDiff-X

### 1. Unsupervised Industrial Anomaly Detection

Key idea:

- train only on normal data
- detect anomalies by reconstruction, density mismatch, feature inconsistency, or patch-level deviation

Why it matters here:

- the challenge setup is naturally unsupervised
- MulSen-AD training folders contain normal-only data
- evaluation is performed on normal and defective test samples with GT

### 2. Diffusion Models for Anomaly Detection

Key idea:

- diffusion models learn normal data distributions through denoising
- anomalies can be exposed through poor reconstruction, poor denoising consistency, or low conditioned likelihood

Why it matters here:

- diffusion offers a stronger normality prior than many classic autoencoder baselines
- the model can generate appearance-consistent residual signals for localisation

### 3. Multi-Modal and Multi-Sensor Anomaly Detection

Key idea:

- different modalities reveal different failure mechanisms
- fusion can improve robustness when each modality captures complementary information

Why it matters here:

- RGB shows visible surface anomalies
- infrared reveals thermal irregularities
- point clouds reveal geometric and structural deviations

### 4. Physics-Informed or Descriptor-Based Conditioning

Key idea:

- raw multimodal fusion is often expensive and brittle
- engineered or semi-structured descriptors can inject domain evidence in a lighter and more interpretable way

Why it matters here:

- the current project design intentionally transforms infrared and point cloud into descriptor maps and global vectors
- this keeps the diffusion target manageable while preserving sensor meaning

### 5. Explainable Industrial AI

Key idea:

- operators need interpretable evidence, not only anomaly scores
- localisation, sensor attribution, and defect hypotheses improve usability

Why it matters here:

- the challenge framing benefits from explanation beyond binary detection
- descriptor evidence naturally supports grounded explanation

### 6. Retrieval-Grounded Language Generation

Key idea:

- language models are more reliable when grounded in structured evidence and retrieved context
- free-form explanation without evidence packaging is risky

Why it matters here:

- the project should package anomaly evidence first, then optionally use an LLM to produce root-cause narratives and recommended actions

## Project Positioning

MulSenDiff-X sits at the intersection of:

- unsupervised industrial anomaly detection
- descriptor-conditioned diffusion modelling
- multi-sensor evidence fusion
- evidence-grounded explanation

The distinguishing position is not just "multimodal diffusion." It is:

- RGB-driven normality modelling
- descriptor-conditioned use of infrared and geometry
- localisation with cross-modal support
- explanation built from explicit evidence packaging

## Practical Takeaways for Implementation

- do not overcomplicate the first model with full raw multimodal generation
- prioritise stable normal-only learning on RGB
- use auxiliary modalities where they help most: conditioning, support scoring, and explanation
- build evaluation and interpretability into the pipeline from the start

## Citation To-Do Before Paper Writing

Before the final report or paper:

- verify the exact MulSen-AD dataset citation
- add diffusion-anomaly references used for the detector framing
- add multimodal industrial AD references
- add explanation and retrieval-grounding references
- add any challenge-specific benchmark references if required
