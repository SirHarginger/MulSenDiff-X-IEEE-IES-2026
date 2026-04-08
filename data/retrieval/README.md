# Retrieval Index

This directory contains the prebuilt trusted-corpus index used by the public demo app.

Included file:

- `index.jsonl`

The app reads this index directly at runtime for detector-grounded retrieval.

The source reference markdown files used during local development are not included in the public repo. The retained metadata templates live under:

- `docs/references/`

If you want to rebuild the index yourself, add approved `.md` or `.txt` source documents alongside matching `.meta.json` files under `docs/references/`, then run:

```bash
python scripts/build_trusted_corpus.py
```
