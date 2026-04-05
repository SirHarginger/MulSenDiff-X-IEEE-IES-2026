from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REGIME_ALIASES = {
    "shared": ("CCDD", "Category-Conditioned Descriptor Diffusion"),
    "shared_nocat": ("CADD", "Category-Agnostic Descriptor Diffusion"),
    "per_category": ("CSDD", "Category-Specialized Descriptor Diffusion"),
}

REGIME_OUTPUT_DIRS = {
    "CCDD": "ccdd",
    "CADD": "cadd",
    "CSDD": "csdd",
}

REGISTRY_FIELDNAMES = [
    "timestamp",
    "regime_internal",
    "regime_paper",
    "regime_expansion",
    "phase",
    "run_type",
    "scope",
    "category",
    "run_dir",
    "checkpoint_path",
    "data_root",
    "processed_root",
    "git_commit",
    "notes",
]


@dataclass(frozen=True)
class RegimeIdentity:
    internal: str
    paper: str
    expansion: str


def resolve_regime_identity(
    *,
    joint_mode: bool,
    disable_category_embedding: bool = False,
) -> RegimeIdentity:
    if joint_mode and disable_category_embedding:
        internal = "shared_nocat"
    elif joint_mode:
        internal = "shared"
    else:
        internal = "per_category"
    paper, expansion = REGIME_ALIASES[internal]
    return RegimeIdentity(internal=internal, paper=paper, expansion=expansion)


def resolve_processed_root(*, data_root: Path | str = "data", processed_root: Path | str | None = None) -> Path:
    return Path(processed_root) if processed_root is not None else Path(data_root) / "processed"


def processed_root_ready(processed_root: Path | str) -> bool:
    root = Path(processed_root)
    samples_root = root / "samples"
    required_paths = [
        root / "manifests" / "dataset_index.csv",
        root / "reports" / "descriptor_pipeline_summary.json",
        root / "category_stats",
        samples_root,
    ]
    if not all(path.exists() for path in required_paths):
        return False
    if not any(path.is_dir() for path in samples_root.iterdir()):
        return False
    try:
        summary = json.loads((root / "reports" / "descriptor_pipeline_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    generated = summary.get("generated", {})
    return int(generated.get("samples", 0) or 0) > 0


def is_joint_category_request(selected_categories: Sequence[str] | None = None, *, category: str = "") -> bool:
    values = [str(item).strip().lower() for item in (selected_categories or []) if str(item).strip()]
    if any(value == "all" for value in values):
        return True
    if len(values) > 1:
        return True
    return str(category).strip().lower() == "shared"


def resolve_scope(selected_categories: Sequence[str] | None = None, *, category: str = "") -> str:
    values = [item.strip() for item in (selected_categories or []) if str(item).strip()]
    if any(value.lower() == "all" for value in values):
        return "all"
    if len(values) > 1:
        return "all"
    if len(values) == 1:
        return values[0]
    return str(category).strip() or "all"


def resolve_regime_output_dir(regime_paper: str) -> str:
    return REGIME_OUTPUT_DIRS.get(str(regime_paper).strip().upper(), str(regime_paper).strip().lower())


def classify_ablation_family(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if not normalized or normalized == "baseline":
        return ""
    if any(token in normalized for token in ("archetype_a", "descriptor_replacement", "localization", "cp2")):
        return "archetype_a"
    if any(token in normalized for token in ("archetype_b", "internal_defect_gate", "gate_on", "gate_off", "cp3")):
        return "archetype_b"
    return ""


def default_output_root(
    *,
    regime_paper: str,
    run_type: str,
    scope: str,
    label: str | None = None,
    repo_root: Path | str = ".",
) -> Path:
    base = Path(repo_root) / "runs"
    normalized_label = (label or "baseline").strip().lower()
    ablation_family = classify_ablation_family(normalized_label)
    if ablation_family:
        return base / "ablations" / ablation_family / run_type.lower()
    return base / resolve_regime_output_dir(regime_paper) / run_type.lower()


def build_structured_run_name(
    *,
    timestamp: str,
    regime_paper: str,
    scope: str,
    run_type: str,
    label: str,
) -> str:
    return "__".join(
        [
            timestamp,
            regime_paper.lower(),
            scope.lower(),
            run_type.lower(),
            label.lower(),
        ]
    )


def ensure_registry(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDNAMES)
            writer.writeheader()
    return path


def append_registry_rows(path: Path | str, rows: Sequence[Dict[str, Any]]) -> Path:
    path = ensure_registry(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDNAMES)
        for row in rows:
            payload = {field: row.get(field, "") for field in REGISTRY_FIELDNAMES}
            writer.writerow(payload)
    return path


def discover_evaluation_runs(*roots: Path | str) -> List[Path]:
    candidates: List[Path] = []
    for root_like in roots:
        root = Path(root_like)
        if not root.exists():
            continue
        iterator = (path.parent for path in root.rglob("summary.json"))
        for run_root in iterator:
            summary_path = run_root / "summary.json"
            evidence_index_path = run_root / "evidence" / "index.json"
            calibration_path = run_root / "evidence" / "calibration.json"
            if summary_path.exists() and evidence_index_path.exists() and calibration_path.exists():
                candidates.append(run_root)
    deduped = sorted(
        {path.resolve(): path for path in candidates}.values(),
        key=lambda path: str(path),
        reverse=True,
    )
    return deduped


def discover_training_runs(*roots: Path | str) -> List[Path]:
    candidates: List[Path] = []
    for root_like in roots:
        root = Path(root_like)
        if not root.exists():
            continue
        iterator = (
            path.parent
            for path in root.rglob("config.json")
            if (path.parent / "checkpoints" / "best.pt").exists()
        )
        for run_root in iterator:
            config_path = run_root / "config.json"
            best_checkpoint = run_root / "checkpoints" / "best.pt"
            if config_path.exists() and best_checkpoint.exists():
                candidates.append(run_root)
    deduped = sorted(
        {path.resolve(): path for path in candidates}.values(),
        key=lambda path: str(path),
        reverse=True,
    )
    return deduped


def infer_regime_identity_from_payload(payload: Dict[str, Any]) -> RegimeIdentity:
    ablation_mode = str(payload.get("ablation_mode", "")).strip().lower()
    training_mode = str(payload.get("training_mode", "")).strip().lower()
    disable_category_embedding = bool(payload.get("disable_category_embedding", False))
    if ablation_mode == "shared_nocat" or disable_category_embedding:
        return resolve_regime_identity(joint_mode=True, disable_category_embedding=True)
    if ablation_mode == "per_category" or training_mode == "single_category":
        return resolve_regime_identity(joint_mode=False, disable_category_embedding=False)
    return resolve_regime_identity(joint_mode=True, disable_category_embedding=False)


def load_json(path: Path | str) -> Dict[str, Any] | List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
