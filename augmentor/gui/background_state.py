"""Background Swap 탭의 UI 상태 저장/복원과 즉시 미리보기 (Tk/torch 의존 없음)."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from background_swap.color_region import color_mask
from background_swap.composite import mask_overlay
from background_swap.masks import object_core_mask, object_protect_mask
from background_swap.settings import (BackgroundSwapSettings, load_settings,
                                      save_settings)
from core.label import BBox

UI_STATE_PATH = Path.home() / ".config" / "dataset_helper" / "background_swap_ui.json"


def load_ui_state(path: Path = UI_STATE_PATH) -> BackgroundSwapSettings:
    """저장된 UI 설정. 파일이 없거나 손상되면 기본 설정."""
    try:
        return load_settings(Path(path))
    except Exception:
        return BackgroundSwapSettings()


def save_ui_state(s: BackgroundSwapSettings, path: Path = UI_STATE_PATH) -> None:
    """UI 설정 저장. 실패해도 예외를 내지 않는다."""
    try:
        save_settings(s, Path(path))
    except Exception:
        pass


def to_number(text, cast):
    """안전한 숫자 변환 (cast는 int 또는 float). 실패하면 None, 예외 없음."""
    try:
        v = float(str(text).strip())
        if not math.isfinite(v):
            return None
        return int(v) if cast is int else v
    except (ValueError, OverflowError):
        return None


def parse_axis_values(text: str) -> list[str]:
    """`;` 또는 줄바꿈으로 구분된 값 → 공백 제거, 빈 값 제외."""
    return [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]


def format_axis_values(values: list[str]) -> str:
    return "; ".join(str(v) for v in values)


def quick_mask_preview(img_bgr: np.ndarray, boxes: list[BBox],
                       settings: BackgroundSwapSettings) -> np.ndarray:
    """SAM/텍스트 검출 없이 사각형 core + protect + color keep 만으로 만든 오버레이.

    text 방식 keep 항목은 Preview 실행 결과에서만 표시된다.
    """
    h, w = img_bgr.shape[:2]
    core = object_core_mask(h, w, boxes, [None] * len(boxes), settings.small_box_px)
    protect = object_protect_mask(core, boxes, settings.dilate_px,
                                  settings.shadow_extend_ratio)
    keep = np.zeros((h, w), bool)
    for item in settings.keep_items:
        if item.method != "color" or not item.hsv_lower or not item.hsv_upper:
            continue
        keep |= color_mask(img_bgr, item.hsv_lower, item.hsv_upper,
                           item.largest_only, item.convex_hull)
    return mask_overlay(img_bgr, core, protect, keep)
