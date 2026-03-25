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

## Product Explanation Strategy

The product-facing explanation path is now:

1. build a structured evidence payload from the detector
2. optionally attach retrieved manufacturing context
3. send only that grounded evidence to Gemini
4. require a structured response with:
   - `likely_cause`
   - `why_flagged`
   - `recommended_action`
   - `operator_summary`

Templated explanations may still exist internally for tests or debugging, but they are not the product story and should not be presented as the serious explanation layer.

## Output Style

A useful explanation should answer:

- what looks abnormal
- where it is located
- which sensors support the finding
- what defect family is most plausible
- what action should follow

## Example Structured Output

Example Gemini output shape:

- likely cause: localized surface damage or foreign-body contamination
- why flagged: concentrated anomaly response with supporting thermal/geometric evidence in the dominant region
- recommended action: manual review of the highlighted area and targeted QA recheck
- operator summary: compact operator-facing recap of the finding

## Guardrails

The explanation layer should:

- avoid claims unsupported by descriptor evidence
- distinguish observation from hypothesis
- mark low-confidence cases clearly
- remain category-aware
- abstain cleanly if Gemini is unavailable rather than substituting a product-facing scripted explanation

This keeps the explanation grounded and challenge-appropriate.
