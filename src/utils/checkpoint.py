from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int,
    metrics: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
