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


def apply_flip(
    img: np.ndarray,
    boxes: List[BBox],
    horizontal: bool,
    vertical: bool,
) -> Tuple[np.ndarray, List[BBox]]:
    if horizontal:
        img = cv2.flip(img, 1)
    if vertical:
        img = cv2.flip(img, 0)
    new_boxes = []
    for b in boxes:
        x = (1.0 - b.x) if horizontal else b.x
        y = (1.0 - b.y) if vertical else b.y
        new_boxes.append(BBox(b.cls, x, y, b.w, b.h))
    return img, new_boxes


def apply_crop(
    img: np.ndarray,
    boxes: List[BBox],
    crop_ratio: float,
) -> Tuple[np.ndarray, List[BBox]]:
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    ch, cw = max(ch, 1), max(cw, 1)
    x_off = random.randint(0, w - cw)
    y_off = random.randint(0, h - ch)

    cropped = img[y_off:y_off + ch, x_off:x_off + cw]
    out = cv2.resize(cropped, (w, h))

    rx1 = x_off / w
    ry1 = y_off / h
    rx2 = (x_off + cw) / w
    ry2 = (y_off + ch) / h

    new_boxes = []
    for b in boxes:
        bx1 = max(b.x - b.w / 2, rx1)
        by1 = max(b.y - b.h / 2, ry1)
        bx2 = min(b.x + b.w / 2, rx2)
        by2 = min(b.y + b.h / 2, ry2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        orig_area = b.w * b.h
        if orig_area > 0 and (bx2 - bx1) * (by2 - by1) / orig_area < 0.3:
            continue
        nx1 = (bx1 - rx1) / crop_ratio
        ny1 = (by1 - ry1) / crop_ratio
        nx2 = (bx2 - rx1) / crop_ratio
        ny2 = (by2 - ry1) / crop_ratio
        new_boxes.append(BBox(b.cls, (nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1))
    return out, new_boxes


def apply_rotation(
    img: np.ndarray,
    boxes: List[BBox],
    angle: float,
) -> Tuple[np.ndarray, List[BBox]]:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    new_boxes = []
    for b in boxes:
        bx1 = (b.x - b.w / 2) * w
        by1 = (b.y - b.h / 2) * h
        bx2 = (b.x + b.w / 2) * w
        by2 = (b.y + b.h / 2) * h
        corners = np.array(
            [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32
        )
        ones = np.ones((4, 1), dtype=np.float32)
        rotated_corners = (M @ np.hstack([corners, ones]).T).T
        rx1 = float(np.clip(rotated_corners[:, 0].min() / w, 0.0, 1.0))
        ry1 = float(np.clip(rotated_corners[:, 1].min() / h, 0.0, 1.0))
        rx2 = float(np.clip(rotated_corners[:, 0].max() / w, 0.0, 1.0))
        ry2 = float(np.clip(rotated_corners[:, 1].max() / h, 0.0, 1.0))
        nw, nh = rx2 - rx1, ry2 - ry1
        if nw > 0 and nh > 0:
            new_boxes.append(BBox(b.cls, (rx1 + rx2) / 2, (ry1 + ry2) / 2, nw, nh))
    return rotated, new_boxes
