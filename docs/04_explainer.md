# Evidence-Grounded Explanation Module

## Goal

The explanation module should not invent causes directly from raw model scores. It should convert anomaly outputs into a structured evidence bundle first, then generate a grounded explanation from that bundle.

## Inputs

The explainer should consume:

- category name
- sample identifier
- image-level anomaly score
- localisation map
- dominant anomaly region summary
- infrared descriptor evidence
- point-cloud descriptor evidence
- cross-modal agreement indicators
- optional retrieved manufacturing knowledge

## Evidence Package

A first structured evidence payload can contain:

- `category`
- `sample_id`
- `score`
- `top_regions`
- `rgb_observations`
- `thermal_observations`
- `geometric_observations`
- `cross_modal_support`
- `confidence_notes`
- `retrieved_context`

This package should be serialisable and inspectable before any language generation happens.

## First-Stage Explanation Strategy

The safest initial version is evidence templating:

- identify the strongest anomaly region
- summarise the RGB residual pattern
- summarise thermal evidence
- summarise geometric evidence
- state whether the modalities agree
- propose a cautious root-cause hypothesis
- suggest a next inspection action

This gives reliable, debuggable outputs even before an LLM is connected.

## Later LLM Integration

When the detector is stable, the explainer can add:

- retrieval over a small manufacturing knowledge base
- prompt templates driven by structured evidence
- grounded recommendation generation

The LLM should receive only evidence that has already been computed and validated by the detector pipeline.

## Output Style

A useful explanation should answer:

- what looks abnormal
- where it is located
- which sensors support the finding
- what defect family is most plausible
- what action should follow

## Example Explanation Skeleton

Example output shape:

- anomaly detected in the upper-right region of the part
- RGB residuals indicate appearance inconsistency
- infrared descriptors indicate a local thermal hotspot
- point-cloud descriptors indicate local curvature or surface deviation
- cross-modal agreement is strong
- plausible cause: surface damage, foreign body, or structural deformation depending on category
- recommended action: manual inspection or targeted QA review

## Guardrails

The explanation layer should:

- avoid claims unsupported by descriptor evidence
- distinguish observation from hypothesis
- mark low-confidence cases clearly
- remain category-aware

This keeps the explanation grounded and challenge-appropriate.
