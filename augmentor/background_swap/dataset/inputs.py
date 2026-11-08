"""입력 이미지 선택, 스킵 사유 판정, 학습 전용 가드 헬퍼."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from background_swap.common.domain import ImageItem
from core.pipeline import _collect_originals

NON_TRAIN_PARTS = {"val", "valid", "validation", "test"}


def looks_non_train(path: Path) -> bool:
    return any(part.lower() in NON_TRAIN_PARTS for part in Path(path).parts)


def load_image_bgr(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def collect_items(
    images_dir: Path, labels_dir: Path, frame_stride: int
) -> tuple[list[ImageItem], list[tuple[str, str]]]:
    selected = _collect_originals(images_dir)[::frame_stride]
    items: list[ImageItem] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in selected:
        stem = path.stem
        if stem in seen:
            skipped.append((stem, "duplicate_stem"))
            continue
        seen.add(stem)
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            skipped.append((stem, "missing_label"))
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            skipped.append((stem, "unreadable_image"))
            continue
        items.append(ImageItem(stem, path, label_path, width, height))
    return items, skipped
