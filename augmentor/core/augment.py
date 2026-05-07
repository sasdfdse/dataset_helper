import random
import cv2
import numpy as np
from typing import List, Tuple

from core.label import BBox, transform_bboxes_scale, filter_bboxes_cutmix_a, transform_bboxes_cutmix_b


def apply_brightness(img: np.ndarray, delta: int) -> np.ndarray:
    result = img.astype(np.int32) + delta
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def apply_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius % 2 == 0:
        k = radius + 1
    else:
        k = radius
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_noise(img: np.ndarray, strength: float) -> np.ndarray:
    noise = np.random.normal(0, strength, img.shape).astype(np.int32)
    result = img.astype(np.int32) + noise
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def apply_scale(
    img: np.ndarray,
    boxes: List[BBox],
    scale: float,
) -> Tuple[np.ndarray, List[BBox]]:
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    if scale > 1.0:
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        out = resized[y1:y1 + h, x1:x1 + w]
    else:
        out = np.zeros_like(img)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        out[y1:y1 + new_h, x1:x1 + new_w] = resized

    new_boxes = transform_bboxes_scale(boxes, scale)
    return out, new_boxes


def apply_cutmix(
    img_a: np.ndarray,
    boxes_a: List[BBox],
    img_b: np.ndarray,
    boxes_b: List[BBox],
) -> Tuple[np.ndarray, List[BBox]]:
    h, w = img_a.shape[:2]

    rx1 = random.uniform(0.1, 0.5)
    ry1 = random.uniform(0.1, 0.5)
    rx2 = random.uniform(rx1 + 0.2, min(rx1 + 0.6, 0.9))
    ry2 = random.uniform(ry1 + 0.2, min(ry1 + 0.6, 0.9))

    px1, py1 = int(rx1 * w), int(ry1 * h)
    px2, py2 = int(rx2 * w), int(ry2 * h)

    result = img_a.copy()
    patch = cv2.resize(img_b, (px2 - px1, py2 - py1))
    result[py1:py2, px1:px2] = patch

    kept_a = filter_bboxes_cutmix_a(boxes_a, rx1, ry1, rx2, ry2)
    new_b = transform_bboxes_cutmix_b(boxes_b, rx1, ry1, rx2, ry2)

    return result, kept_a + new_b
