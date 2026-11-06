"""합성 / 차이 이미지 / 마스크 오버레이 (torch 의존 없음 — GUI도 사용)."""
from __future__ import annotations

import cv2
import numpy as np

_RED = (0, 0, 255)
_ORANGE = (0, 140, 255)
_GREEN = (0, 200, 0)


def alpha_map(protect: np.ndarray, feather_px: int) -> np.ndarray:
    """원본 비중 알파 (float32). protect 내부 1.0, 바깥은 거리에 따라 선형 감쇠."""
    protect = protect.astype(bool)
    if feather_px <= 0 or not protect.any():
        return protect.astype(np.float32)
    dist = cv2.distanceTransform((~protect).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.maximum(0.0, 1.0 - dist / (feather_px + 1)).astype(np.float32)
    alpha[protect] = 1.0
    return alpha


def composite(orig: np.ndarray, gen: np.ndarray, protect: np.ndarray,
              feather_px: int) -> np.ndarray:
    """생성 배경을 원본 위에 합성. protect 내부는 원본 바이트 그대로."""
    protect = protect.astype(bool)
    alpha = alpha_map(protect, feather_px)[..., None]
    blend = orig.astype(np.float32) * alpha + gen.astype(np.float32) * (1.0 - alpha)
    out = np.clip(np.rint(blend), 0, 255).astype(np.uint8)
    out[protect] = orig[protect]
    return out


def diff_image(orig: np.ndarray, out: np.ndarray, protect: np.ndarray) -> np.ndarray:
    """채널 최대 절대차 그레이(BGR 3채널). protect 내부 변경 픽셀은 빨강."""
    diff = np.abs(out.astype(np.int16) - orig.astype(np.int16)).max(axis=2).astype(np.uint8)
    img = np.repeat(diff[..., None], 3, axis=2)
    img[protect.astype(bool) & (diff > 0)] = _RED
    return img


def mask_overlay(img_bgr: np.ndarray, object_core: np.ndarray, object_protect: np.ndarray,
                 keep: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """core(빨강) > protect 링(주황) > keep(초록) 순으로 반투명 오버레이."""
    core = object_core.astype(bool)
    ring = object_protect.astype(bool) & ~core
    keep_only = keep.astype(bool) & ~core & ~ring
    colored = img_bgr.astype(np.float32)
    for mask, color in ((core, _RED), (ring, _ORANGE), (keep_only, _GREEN)):
        colored[mask] = (1.0 - alpha) * colored[mask] + alpha * np.array(color, np.float32)
    return np.clip(np.rint(colored), 0, 255).astype(np.uint8)
