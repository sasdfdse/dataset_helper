"""Background Swap 공용 도메인 타입 (torch 의존 없음)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ACCEPTED = "accepted"
REJECTED = "rejected"
SKIPPED = "skipped"
EXHAUSTED = "exhausted"


class GpuOutOfMemory(RuntimeError):
    """GPU 메모리 부족 (모델/해상도 폴백 신호)."""


@dataclass(frozen=True)
class ImageItem:
    stem: str
    image_path: Path
    label_path: Path
    width: int
    height: int


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # pixel xyxy
    score: float
    label: str


@dataclass
class MaskSet:
    """모든 마스크는 HxW bool."""
    object_core: np.ndarray
    object_protect: np.ndarray
    keep: np.ndarray
    inpaint: np.ndarray
    warnings: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def protect(self) -> np.ndarray:
        return self.object_protect | self.keep


@dataclass
class Job:
    stem: str
    variant: int
    attempt: int
    seed: int
    scene: dict[str, str]
    prompt: str
    negative_prompt: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scene"] = dict(self.scene)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            stem=d["stem"],
            variant=int(d["variant"]),
            attempt=int(d["attempt"]),
            seed=int(d["seed"]),
            scene=dict(d["scene"]),
            prompt=d["prompt"],
            negative_prompt=d["negative_prompt"],
        )


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, object] = field(default_factory=dict)
