"""합성 / 차이 이미지 / 마스크 오버레이 (torch 의존 없음 — GUI도 사용)."""
from __future__ import annotations

import cv2
import numpy as np

from core.label import BBox

_RED = (0, 0, 255)
_ORANGE = (0, 140, 255)
_GREEN = (0, 200, 0)

# Same values as _BBOX_COLORS in gui/augment_tab.py (copied to avoid importing the GUI)
_BBOX_COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44",
    "#FF44FF", "#44FFFF", "#FF8844", "#44FF88",
]


def draw_boxes(img_bgr: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    """정규화 bbox를 클래스별 색 2px 사각형으로 그린 복사본 (AugmentTab._draw_bboxes와 같은 방식)."""
    result = img_bgr.copy()
    h, w = result.shape[:2]
    for b in boxes:
        color_hex = _BBOX_COLORS[b.cls % len(_BBOX_COLORS)]
        color_bgr = tuple(int(color_hex[j:j + 2], 16) for j in (5, 3, 1))
        x1 = int((b.x - b.w / 2) * w)
        y1 = int((b.y - b.h / 2) * h)
        x2 = int((b.x + b.w / 2) * w)
        y2 = int((b.y + b.h / 2) * h)
        cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 2)
    return result


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
