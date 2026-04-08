from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.explainer.evidence_builder import EvidencePackage, build_retrieval_query

ALLOWED_TRUST_LEVELS = {"approved", "reference"}


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: str
    source_type: str
    trust_level: str
    category_tags: List[str]
    sensor_tags: List[str]
    defect_tags: List[str]
    process_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedContextItem:
    title: str
    snippet: str
    source: str
    score: float
    relevance_reason: str = ""
    kind: str = "trusted_corpus_chunk"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", str(text).lower())
        if len(token) >= 3
    }


def resolve_corpus_index_path(knowledge_base_root: Path | str | None) -> Path | None:
    if knowledge_base_root is None:
        return None
    root = Path(knowledge_base_root)
    if not root.exists():
        return None
    if root.is_file() and root.name == "index.jsonl":
        return root

    candidates = [
        root / "index.jsonl",
        root / "data" / "retrieval" / "index.jsonl",
        root / "trusted_corpus" / "index.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_trusted_corpus(knowledge_base_root: Path | str | None) -> List[CorpusChunk]:
    index_path = resolve_corpus_index_path(knowledge_base_root)
    if index_path is None:
        return []

    chunks: List[CorpusChunk] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        chunks.append(
            CorpusChunk(
                chunk_id=str(payload["chunk_id"]),
                doc_id=str(payload["doc_id"]),
                title=str(payload["title"]),
                text=str(payload["text"]),
                source_path=str(payload["source_path"]),
                source_type=str(payload["source_type"]),
                trust_level=str(payload["trust_level"]).lower(),
                category_tags=_normalize_tag_list(payload.get("category_tags", [])),
                sensor_tags=_normalize_tag_list(payload.get("sensor_tags", [])),
                defect_tags=_normalize_tag_list(payload.get("defect_tags", [])),
                process_tags=_normalize_tag_list(payload.get("process_tags", [])),
            )
        )
    return chunks


def retrieve_context_for_evidence(
    package: EvidencePackage,
    *,
    knowledge_base_root: Path | str | None = None,
    top_k: int = 3,
) -> List[RetrievedContextItem]:
    chunks = load_trusted_corpus(knowledge_base_root)
    if not chunks:
        return []

    query = build_retrieval_query(package)
    query_text_tokens = _tokenize(str(query.get("query_text", "")))
    category = str(query.get("category", "")).strip().lower()
    sensor_tags = set(_normalize_tag_list(query.get("sensor_profile_tags", [])))
    defect_tags = set(
        _normalize_tag_list(query.get("defect_profile_tags", []))
        + _normalize_tag_list(query.get("dominant_modalities", []))
    )
    metadata_tokens = set([category, *sensor_tags, *defect_tags])
    recall_tokens = query_text_tokens | metadata_tokens

    candidates: List[RetrievedContextItem] = []
    for chunk in chunks:
        if chunk.trust_level not in ALLOWED_TRUST_LEVELS:
            continue
        if category and chunk.category_tags and category not in set(chunk.category_tags):
            continue

        chunk_tokens = _chunk_recall_tokens(chunk)
        overlap = recall_tokens & chunk_tokens
        if not overlap:
            continue

        category_score = _category_score(category, chunk.category_tags)
        sensor_profile_score = _overlap_ratio(sensor_tags, set(chunk.sensor_tags))
        evidence_compatibility_score = _overlap_ratio(
            defect_tags,
            set(chunk.defect_tags) | set(chunk.process_tags),
        )
        source_trust_score = 1.0 if chunk.trust_level == "approved" else 0.7
        rerank_score = (
            0.40 * category_score
            + 0.25 * sensor_profile_score
            + 0.20 * evidence_compatibility_score
            + 0.15 * source_trust_score
        )
        if rerank_score <= 0.0:
            continue

        candidates.append(
            RetrievedContextItem(
                title=chunk.title,
                snippet=_compact_snippet(chunk.text),
                source=chunk.source_path,
                score=round(float(rerank_score), 6),
                relevance_reason=_build_relevance_reason(
                    category_score=category_score,
                    sensor_profile_score=sensor_profile_score,
                    evidence_compatibility_score=evidence_compatibility_score,
                    source_trust_score=source_trust_score,
                    trust_level=chunk.trust_level,
                ),
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "trust_level": chunk.trust_level,
                    "category_tags": list(chunk.category_tags),
                    "sensor_tags": list(chunk.sensor_tags),
                    "defect_tags": list(chunk.defect_tags),
                    "process_tags": list(chunk.process_tags),
                    "category_score": round(category_score, 6),
                    "sensor_profile_score": round(sensor_profile_score, 6),
                    "evidence_compatibility_score": round(evidence_compatibility_score, 6),
                    "source_trust_score": round(source_trust_score, 6),
                },
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    deduped: List[RetrievedContextItem] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in candidates:
        key = (item.title, item.source)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
        if len(deduped) >= (max(top_k, 0) or 3):
            break
    return deduped


def retrieve_reference_context_for_package(
    package: EvidencePackage,
    *,
    eval_run_root: Path | str | None = None,
    top_k: int = 3,
) -> List[RetrievedContextItem]:
    del package, eval_run_root, top_k
    return []


def build_corpus_index(
    *,
    source_root: Path | str,
    max_chunk_chars: int = 1200,
) -> List[CorpusChunk]:
    root = Path(source_root)
    if not root.exists():
        return []

    chunks: List[CorpusChunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        if path.name.endswith(".meta.json"):
            continue

        metadata_path = path.with_name(f"{path.stem}.meta.json")
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _metadata_is_valid(metadata):
            continue

        doc_id = str(metadata["doc_id"])
        title = str(metadata["title"])
        source_type = str(metadata["source_type"])
        trust_level = str(metadata["trust_level"]).lower()
        text = _load_source_text(path)
        sections = _split_sections(text, suffix=path.suffix.lower())
        merged_sections = _merge_sections(sections, max_chunk_chars=max_chunk_chars)
        for index, section in enumerate(merged_sections, start=1):
            if not section.strip():
                continue
            chunks.append(
                CorpusChunk(
                    chunk_id=f"{doc_id}::{index:04d}",
                    doc_id=doc_id,
                    title=title,
                    text=section.strip(),
                    source_path=str(path),
                    source_type=source_type,
                    trust_level=trust_level,
                    category_tags=_normalize_tag_list(metadata.get("category_tags", [])),
                    sensor_tags=_normalize_tag_list(metadata.get("sensor_tags", [])),
                    defect_tags=_normalize_tag_list(metadata.get("defect_tags", [])),
                    process_tags=_normalize_tag_list(metadata.get("process_tags", [])),
                )
            )
    return chunks


def write_corpus_index(chunks: Sequence[CorpusChunk], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(chunk.to_dict(), sort_keys=True) for chunk in chunks]
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def _normalize_tag_list(values: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values:
        token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _chunk_recall_tokens(chunk: CorpusChunk) -> set[str]:
    return (
        _tokenize(chunk.text)
        | _tokenize(chunk.title)
        | set(chunk.category_tags)
        | set(chunk.sensor_tags)
        | set(chunk.defect_tags)
        | set(chunk.process_tags)
    )


def _category_score(category: str, category_tags: Sequence[str]) -> float:
    normalized = {tag.lower() for tag in category_tags}
    if category and category in normalized:
        return 1.0
    if not normalized or normalized & {"global", "generic", "all"}:
        return 0.5
    return 0.0


def _overlap_ratio(query_values: set[str], candidate_values: set[str]) -> float:
    if not query_values or not candidate_values:
        return 0.0
    return float(len(query_values & candidate_values)) / float(len(query_values))


def _build_relevance_reason(
    *,
    category_score: float,
    sensor_profile_score: float,
    evidence_compatibility_score: float,
    source_trust_score: float,
    trust_level: str,
) -> str:
    reasons: List[str] = []
    if category_score >= 1.0:
        reasons.append("exact category match")
    elif category_score > 0.0:
        reasons.append("global guidance match")
    if sensor_profile_score > 0.0:
        reasons.append("sensor-profile overlap")
    if evidence_compatibility_score > 0.0:
        reasons.append("evidence-compatible defect profile")
    reasons.append(f"trust={trust_level}")
    reasons.append(f"trust_score={source_trust_score:.2f}")
    return "; ".join(reasons)


def _compact_snippet(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    return compact[:limit]


def _metadata_is_valid(metadata: Mapping[str, Any]) -> bool:
    required = [
        "doc_id",
        "title",
        "source_type",
        "trust_level",
        "category_tags",
        "sensor_tags",
        "defect_tags",
        "process_tags",
    ]
    if any(key not in metadata for key in required):
        return False
    return str(metadata.get("trust_level", "")).lower() in ALLOWED_TRUST_LEVELS


def _load_source_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(payload, indent=2, sort_keys=True)
    return path.read_text(encoding="utf-8")


def _split_sections(text: str, *, suffix: str) -> List[str]:
    cleaned = str(text).replace("\r\n", "\n")
    if suffix == ".md":
        parts = re.split(r"(?m)(?=^#{1,6}\s)", cleaned)
    else:
        parts = re.split(r"\n\s*\n", cleaned)
    sections = [part.strip() for part in parts if part.strip()]
    return sections or [cleaned.strip()]


def _merge_sections(sections: Iterable[str], *, max_chunk_chars: int) -> List[str]:
    merged: List[str] = []
    buffer = ""
    for section in sections:
        candidate = section.strip()
        if not candidate:
            continue
        if len(candidate) > max_chunk_chars:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.extend(_hard_split(candidate, max_chunk_chars=max_chunk_chars))
            continue
        if not buffer:
            buffer = candidate
            continue
        combined = f"{buffer}\n\n{candidate}"
        if len(combined) <= max_chunk_chars:
            buffer = combined
        else:
            merged.append(buffer.strip())
            buffer = candidate
    if buffer:
        merged.append(buffer.strip())
    return merged


def _hard_split(text: str, *, max_chunk_chars: int) -> List[str]:
    paragraphs = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = paragraph.strip()
        if not candidate:
            continue
        if len(candidate) > max_chunk_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            for start in range(0, len(candidate), max_chunk_chars):
                chunks.append(candidate[start : start + max_chunk_chars].strip())
            continue
        combined = f"{buffer} {candidate}".strip() if buffer else candidate
        if len(combined) <= max_chunk_chars:
            buffer = combined
        else:
            chunks.append(buffer.strip())
            buffer = candidate
    if buffer:
        chunks.append(buffer.strip())
    return chunks
