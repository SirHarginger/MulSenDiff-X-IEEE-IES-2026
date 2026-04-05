#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainer.retriever import build_corpus_index, write_corpus_index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the curated trusted-corpus chunk index for detector-first corrective RAG.",
    )
    parser.add_argument("--source-root", default="docs/references")
    parser.add_argument("--output", default="data/retrieval/index.jsonl")
    parser.add_argument("--max-chars", type=int, default=1200)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    source_root = Path(args.source_root)
    output_path = Path(args.output)

    chunks = build_corpus_index(
        source_root=source_root,
        max_chunk_chars=max(int(args.max_chars), 200),
    )
    output_path = write_corpus_index(chunks, output_path)
    print(f"source_root: {source_root}")
    print(f"output_path: {output_path}")
    print(f"chunks_written: {len(chunks)}")


if __name__ == "__main__":
    main()
