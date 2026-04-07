# Detector-To-Explainer Flow

## Goal

The explanation path exists to interpret detector evidence for an operator. It does **not** define the anomaly decision and it is **not** part of the core detector benchmark.

## Current Product Logic

The product-facing explanation flow is:

1. MulSenDiff-X detects and localizes an anomaly.
2. The detector exports a structured evidence payload.
3. Gemini optionally converts that grounded payload into an operator-facing explanation.

The detector remains the source of truth.

## Evidence Package

The explanation input is a structured evidence package built from detector outputs such as:

- category
- sample identifier
- anomaly score
- status / severity / confidence
- affected area
- top regions
- RGB observations
- thermal observations
- geometric observations
- cross-modal support
- confidence notes
- provenance and source paths

This package should be inspectable before any language generation happens.

## Gemini’s Role

Gemini is:

- optional at runtime
- product-facing
- downstream of detection

Gemini is not:

- part of detector scoring
- part of calibration
- part of the benchmark metrics

The intended structured Gemini output remains:

- `likely_cause`
- `why_flagged`
- `recommended_action`
- `operator_summary`

## Benchmark Position

Explanation quality should not be reported as a core detector result unless there is a real human evaluation protocol.

That means:

- no explanation metrics in the main detector benchmark by default
- no mixing detector gains with explanation gains in the same claim
- qualitative explanation examples are fine
- quantitative explanation results need:
  - a fixed sample set
  - a rating rubric
  - at least two human raters
  - inter-rater agreement

## Guardrails

The explanation layer should:

- use only detector-grounded evidence
- avoid unsupported causal claims
- distinguish observation from hypothesis
- express uncertainty when evidence is mixed
- fail clearly if Gemini is unavailable rather than pretending to be an LLM with scripted text

## App Note

The current app is a paired-sample demo. The explanation it shows is only as trustworthy as the detector evidence produced for that matched sample.
