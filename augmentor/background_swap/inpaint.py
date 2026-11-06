"""인페인터 프로토콜, 작업 해상도 규칙, 인페인트 실행 (torch 의존 없음)."""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .composite import composite
from .domain import Job, MaskSet
from .settings import BackgroundSwapSettings


class Inpainter(Protocol):
    native_px: int

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str,
                negative_prompt: str, seed: int) -> np.ndarray:
        """입력 크기 그대로 RGB 이미지를 반환한다. mask True = 다시 그릴 영역."""
        ...


def working_size(w: int, h: int, native_px: int, max_px: int, mode: str,
                 manual_px: int) -> tuple[int, int]:
    """(width, height). 긴 변 목표에 맞춰 비율 유지, 각 변은 8의 배수(최소 8)."""
    long_side = max(w, h)
    target = manual_px if mode == "manual" else min(max(long_side, native_px), max_px)
    scale = target / long_side
    return (max(8, round(w * scale / 8) * 8), max(8, round(h * scale / 8) * 8))


def run_inpaint(inpainter: Inpainter, orig_bgr: np.ndarray, masks: MaskSet, job: Job,
                settings: BackgroundSwapSettings) -> np.ndarray:
    """작업 해상도에서 인페인트 후 원본 크기로 되돌려 합성한다 (BGR)."""
    h, w = orig_bgr.shape[:2]
    ww, wh = working_size(w, h, inpainter.native_px, settings.max_resolution_px,
                          settings.resolution_mode, settings.manual_resolution_px)
    shrinking = ww * wh < w * h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    small = cv2.resize(orig_bgr, (ww, wh), interpolation=interp)
    mask = cv2.resize(masks.inpaint.astype(np.uint8), (ww, wh),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    gen_rgb = inpainter.inpaint(rgb, mask, job.prompt, job.negative_prompt, job.seed)
    gen = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR)
    if gen.shape[:2] != (h, w):
        back = cv2.INTER_AREA if gen.shape[0] * gen.shape[1] > w * h else cv2.INTER_CUBIC
        gen = cv2.resize(gen, (w, h), interpolation=back)
    return composite(orig_bgr, gen, masks.protect, settings.feather_px)
