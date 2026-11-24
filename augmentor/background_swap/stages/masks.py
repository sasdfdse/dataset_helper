"""객체(core/protect), keep, inpaint 마스크 생성 (torch 비의존).

모든 마스크는 원본 해상도의 HxW bool. 세그멘터/검출기는 Protocol로만 의존한다.
"""
from __future__ import annotations

import math
from typing import Protocol

import cv2
import numpy as np

from core.label import BBox

from background_swap.stages.color_region import color_mask
from background_swap.common.domain import Detection, MaskSet
from background_swap.common.settings import BackgroundSwapSettings, KeepItem

_RECT_MARGIN = 0.1
_SAM_MIN_AREA_RATIO = 0.2
_SHADOW_HALF_WIDTH_RATIO = 0.6
_TEXT_BOX_THRESHOLD = 0.35
_TEXT_TEXT_THRESHOLD = 0.25


class Segmenter(Protocol):
    def segment_boxes(self, image_rgb: np.ndarray,
                      boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        ...


class TextDetector(Protocol):
    def detect(self, image_rgb: np.ndarray, phrases: list[str],
               box_threshold: float, text_threshold: float) -> list[Detection]:
        ...


def _clip_slices(left: float, top: float, right: float, bottom: float,
                 w: int, h: int) -> tuple[int, int, int, int] | None:
    # round first so float error (e.g. 45.00000000000001) doesn't shift ceil/floor
    x1 = max(0, math.floor(round(left, 6)))
    y1 = max(0, math.floor(round(top, 6)))
    x2 = min(w, math.ceil(round(right, 6)))
    y2 = min(h, math.ceil(round(bottom, 6)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_to_xyxy(b: BBox, w: int, h: int,
                 margin: float = 0.0) -> tuple[int, int, int, int] | None:
    """정규화 중심 박스 → 픽셀 반개구간 (x1, y1, x2, y2). `mask[y1:y2, x1:x2]`용.

    각 변을 박스 폭(높이)×margin 만큼 확장하고 이미지로 클립. 폭/높이 0이면 None.
    """
    bw, bh = b.w * w, b.h * h
    if bw <= 0 or bh <= 0:
        return None
    left = b.x * w - bw / 2 - margin * bw
    right = b.x * w + bw / 2 + margin * bw
    top = b.y * h - bh / 2 - margin * bh
    bottom = b.y * h + bh / 2 + margin * bh
    return _clip_slices(left, top, right, bottom, w, h)


def needs_sam(b: BBox, w: int, h: int, small_box_px: int) -> bool:
    return min(b.w * w, b.h * h) >= small_box_px


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def object_core_mask(h: int, w: int, boxes: list[BBox],
                     sam_masks: list[np.ndarray | None],
                     small_box_px: int) -> np.ndarray:
    """박스별 core. SAM 마스크가 있고 충분히 크면 사용, 아니면 여유 10% 사각형 폴백."""
    core = np.zeros((h, w), bool)
    for b, sam in zip(boxes, sam_masks):
        rect = bbox_to_xyxy(b, w, h, margin=_RECT_MARGIN)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        if sam is not None and needs_sam(b, w, h, small_box_px):
            clipped = np.zeros((h, w), bool)
            clipped[y1:y2, x1:x2] = sam[y1:y2, x1:x2]
            clipped = _largest_component(clipped)
            if clipped.sum() >= b.w * w * b.h * h * _SAM_MIN_AREA_RATIO:
                core |= clipped
                continue
        core[y1:y2, x1:x2] = True
    return core


def object_protect_mask(core: np.ndarray, boxes: list[BBox],
                        dilate_px: int, shadow_extend_ratio: float) -> np.ndarray:
    """core 팽창(후광) ∪ 박스 하단 그림자 사각형."""
    h, w = core.shape
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        protect = cv2.dilate(core.astype(np.uint8), k) > 0
    else:
        protect = core.copy()
    if shadow_extend_ratio > 0:
        for b in boxes:
            bw, bh = b.w * w, b.h * h
            if bw <= 0 or bh <= 0:
                continue
            cx, bottom = b.x * w, b.y * h + bh / 2
            rect = _clip_slices(cx - _SHADOW_HALF_WIDTH_RATIO * bw, bottom,
                                cx + _SHADOW_HALF_WIDTH_RATIO * bw,
                                bottom + shadow_extend_ratio * bh, w, h)
            if rect is not None:
                x1, y1, x2, y2 = rect
                protect[y1:y2, x1:x2] = True
    return protect


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def keep_item_mask(item: KeepItem, img_bgr: np.ndarray,
                   segmenter: Segmenter | None,
                   detector: TextDetector | None,
                   *, _rgb: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """(마스크, 성공). 실패 시 마스크는 전부 False."""
    h, w = img_bgr.shape[:2]
    empty = np.zeros((h, w), bool)
    if item.method == "color":
        if item.hsv_lower is None or item.hsv_upper is None:
            return empty, False
        m = color_mask(img_bgr, item.hsv_lower, item.hsv_upper,
                       item.largest_only, item.convex_hull)
        if m.sum() / (h * w) >= item.min_area_ratio and m.any():
            return m, True
        return empty, False
    if item.method == "text":
        if detector is None or segmenter is None:
            return empty, False
        rgb = _rgb if _rgb is not None else _bgr_to_rgb(img_bgr)
        dets = detector.detect(rgb, [item.prompt], _TEXT_BOX_THRESHOLD, _TEXT_TEXT_THRESHOLD)
        if not dets:
            return empty, False
        masks = segmenter.segment_boxes(rgb, [tuple(d.box) for d in dets])
        union = empty.copy()
        for m in masks:
            union |= m.astype(bool)
        return union, True
    return empty, False


def build_mask_set(img_bgr: np.ndarray, boxes: list[BBox],
                   settings: BackgroundSwapSettings,
                   segmenter: Segmenter | None,
                   detector: TextDetector | None) -> MaskSet:
    h, w = img_bgr.shape[:2]
    rgb = _bgr_to_rgb(img_bgr)

    sam_masks: list[np.ndarray | None] = [None] * len(boxes)
    sam_idx = [i for i, b in enumerate(boxes)
               if needs_sam(b, w, h, settings.small_box_px)
               and bbox_to_xyxy(b, w, h, 0.0) is not None]
    if sam_idx and segmenter is not None:
        xyxy = []
        for i in sam_idx:
            x1, y1, x2, y2 = bbox_to_xyxy(boxes[i], w, h, 0.0)
            xyxy.append((float(x1), float(y1), float(x2), float(y2)))
        for i, m in zip(sam_idx, segmenter.segment_boxes(rgb, xyxy)):
            sam_masks[i] = m.astype(bool)

    core = object_core_mask(h, w, boxes, sam_masks, settings.small_box_px)
    protect = object_protect_mask(core, boxes, settings.dilate_px, settings.shadow_extend_ratio)
    keep = np.zeros((h, w), bool)
    warnings: list[str] = []

    for item in settings.keep_items:
        m, ok = keep_item_mask(item, img_bgr, segmenter, detector, _rgb=rgb)
        if ok:
            keep |= m
        elif item.required:
            return MaskSet(core, protect, keep, ~(protect | keep), warnings,
                           f"required_keep_missing:{item.name}")
        else:
            warnings.append(f"keep_not_found:{item.name}")

    inpaint = ~(protect | keep)
    skip = "small_inpaint_area" if inpaint.mean() < settings.min_inpaint_ratio else None
    return MaskSet(core, protect, keep, inpaint, warnings, skip)
