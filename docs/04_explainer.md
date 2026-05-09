# Detector-To-Explainer Flow

## Goal

The explanation path exists to interpret detector evidence for an operator. It does **not** define the anomaly decision and it is **not** part of the core detector benchmark.

## Current Product Logic

The product-facing explanation flow is:

1. MulSenDiff-X detects and localizes an anomaly.
2. The detector exports a structured evidence payload.
3. trusted retrieval attaches category-aligned reference support.
4. Gemini turns that grounded payload into an operator-facing explanation.

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

- the intended product-facing explanation generator
- downstream of detection and retrieval
- constrained to detector-grounded evidence plus retrieved support

Gemini is not:

- part of detector scoring
- part of calibration
- part of the benchmark metrics

If retrieval or Gemini is unavailable, the app falls back to a detector-grounded explanation card and makes that fallback state explicit in the UI.

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
- preserve trusted citations when retrieval succeeds
- avoid unsupported causal claims
- distinguish observation from hypothesis
- express uncertainty when evidence is mixed
- degrade clearly if Gemini is unavailable rather than pretending a live LLM call succeeded

## App Note

The current app is a curated one-page `CCDD` sample-selector demo. The explanation it shows is only as trustworthy as the detector evidence and trusted retrieval context available for that paired sample.
