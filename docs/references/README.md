# Trusted Corpus Metadata

This public repository keeps the trusted-corpus layer lightweight.

What is retained here:

- one `.meta.json` file per reference profile
- the prebuilt retrieval index at `data/retrieval/index.jsonl`

What is intentionally not retained here:

- the category-specific source markdown documents that were used to build the local trusted corpus during development

This keeps the public repo cleaner for reviewers while preserving:

- the retrieval metadata contract
- the tag vocabulary used by the app
- the prebuilt reviewer-facing index already used by the demo

## Metadata Contract

Each metadata file describes one trusted reference profile and follows this shape:

```json
{
  "doc_id": "capsule_internal_defects_v1",
  "title": "Capsule Internal Defect Guidance",
  "source_type": "manual",
  "trust_level": "approved",
  "category_tags": ["capsule"],
  "sensor_tags": ["cross_inconsistency", "thermal_hotspot"],
  "defect_tags": ["internal_defect_profile", "cross_modal_inconsistency"],
  "process_tags": ["seal_check", "assembly_review"]
}
```

The retained `.meta.json` files are best treated as reviewer-visible retrieval metadata templates.

## Runtime Behavior

The app does not read these metadata files directly at runtime. It reads:

- `data/retrieval/index.jsonl`

So the public demo continues to work even though the category markdown source files are no longer included in `docs/references/`.

## If You Want To Rebuild The Corpus

Add your own approved source documents under `docs/references/` using:

- `<name>.md` or `<name>.txt`
- `<name>.meta.json`

Then rebuild:

```bash
python scripts/build_trusted_corpus.py
```

Use category-aligned tags so retrieval remains consistent with detector evidence.
