# Retrieval Index

This directory stores the built chunk index for corrective retrieval.

Build it with:

```bash
python scripts/build_trusted_corpus.py
```

The generated `index.jsonl` is consumed by the detector-first corrective RAG pipeline.
