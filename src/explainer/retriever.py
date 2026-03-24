from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.explainer.evidence_builder import EvidencePackage


@dataclass(frozen=True)
class RetrievedContextItem:
    title: str
    snippet: str
    source: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) >= 3
    }


def _package_query_terms(package: EvidencePackage) -> set[str]:
    parts: List[str] = [
        package.category,
        package.defect_label,
        package.score_mode,
        *package.rgb_observations,
        *package.thermal_observations,
        *package.geometric_observations,
        *package.cross_modal_support,
    ]
    return _tokenize(" ".join(parts))


def _iter_knowledge_documents(knowledge_base_root: Path) -> Sequence[Path]:
    return sorted(
        path
        for path in knowledge_base_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json"}
    )


def _load_document_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(payload, sort_keys=True)
    return path.read_text(encoding="utf-8")


def retrieve_context_for_evidence(
    package: EvidencePackage,
    *,
    knowledge_base_root: Path | str | None = None,
    top_k: int = 3,
) -> List[RetrievedContextItem]:
    if knowledge_base_root is None:
        return []

    root = Path(knowledge_base_root)
    if not root.exists():
        return []

    query_terms = _package_query_terms(package)
    scored_documents: List[RetrievedContextItem] = []
    for path in _iter_knowledge_documents(root):
        text = _load_document_text(path)
        doc_terms = _tokenize(text)
        overlap = query_terms & doc_terms
        if not overlap:
            continue
        score = float(len(overlap)) / max(len(query_terms), 1)
        snippet = re.sub(r"\s+", " ", text).strip()[:240]
        scored_documents.append(
            RetrievedContextItem(
                title=path.stem,
                snippet=snippet,
                source=str(path),
                score=score,
            )
        )

    scored_documents.sort(key=lambda item: item.score, reverse=True)
    return scored_documents[: max(top_k, 0)]
