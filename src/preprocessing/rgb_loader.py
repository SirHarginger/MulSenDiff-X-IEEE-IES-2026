from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class RGBImageInfo:
    path: str
    mode: str
    width: int
    height: int


def load_rgb_info(path: Path | str) -> RGBImageInfo:
    path = Path(path)
    image = Image.open(path)
    return RGBImageInfo(path=str(path), mode=image.mode, width=image.size[0], height=image.size[1])
