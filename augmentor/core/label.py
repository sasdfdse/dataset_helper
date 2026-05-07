from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class BBox:
    cls: int
    x: float   # normalized center x [0, 1]
    y: float   # normalized center y [0, 1]
    w: float   # normalized width [0, 1]
    h: float   # normalized height [0, 1]


def read_labels(label_path: Path) -> List[BBox]:
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append(BBox(
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ))
    return boxes


def write_labels(label_path: Path, boxes: List[BBox]) -> None:
    with open(label_path, 'w') as f:
        for b in boxes:
            f.write(f"{b.cls} {b.x:.6f} {b.y:.6f} {b.w:.6f} {b.h:.6f}\n")


def clip_bbox(
    box: BBox,
    rx1: float, ry1: float, rx2: float, ry2: float,
    min_ratio: float = 0.3,
) -> Optional[BBox]:
    """Clip box to region. Returns None if remaining area < min_ratio of original."""
    bx1 = box.x - box.w / 2
    by1 = box.y - box.h / 2
    bx2 = box.x + box.w / 2
    by2 = box.y + box.h / 2

    cx1 = max(bx1, rx1)
    cy1 = max(by1, ry1)
    cx2 = min(bx2, rx2)
    cy2 = min(by2, ry2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    orig_area = box.w * box.h
    if orig_area == 0:
        return None

    clipped_area = (cx2 - cx1) * (cy2 - cy1)
    if clipped_area / orig_area < min_ratio:
        return None

    return BBox(box.cls, (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1)


def transform_bboxes_scale(boxes: List[BBox], scale: float) -> List[BBox]:
    """Transform bboxes after scale augmentation.

    Scale > 1.0 (zoom in): resize then center-crop.
      offset = (scale - 1) / 2  (in normalized coords)
      new_center = old_center * scale - offset
      new_size   = old_size * scale

    Scale < 1.0 (zoom out): resize then center-pad.
      offset = (1 - scale) / 2
      new_center = offset + old_center * scale
      new_size   = old_size * scale
    """
    result = []
    if scale > 1.0:
        offset = (scale - 1.0) / 2.0
        for b in boxes:
            new_x = b.x * scale - offset
            new_y = b.y * scale - offset
            new_w = b.w * scale
            new_h = b.h * scale
            clipped = clip_bbox(BBox(b.cls, new_x, new_y, new_w, new_h), 0.0, 0.0, 1.0, 1.0, min_ratio=0.0)
            if clipped is not None:
                result.append(clipped)
    else:
        offset = (1.0 - scale) / 2.0
        for b in boxes:
            new_x = offset + b.x * scale
            new_y = offset + b.y * scale
            new_w = b.w * scale
            new_h = b.h * scale
            result.append(BBox(b.cls, new_x, new_y, new_w, new_h))
    return result


def _intersection_area(b: BBox, rx1: float, ry1: float, rx2: float, ry2: float) -> float:
    bx1 = b.x - b.w / 2
    by1 = b.y - b.h / 2
    bx2 = b.x + b.w / 2
    by2 = b.y + b.h / 2
    ix1 = max(bx1, rx1)
    iy1 = max(by1, ry1)
    ix2 = min(bx2, rx2)
    iy2 = min(by2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def filter_bboxes_cutmix_a(
    boxes: List[BBox],
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> List[BBox]:
    """Keep boxes from image A that are not mostly covered by the CutMix patch.
    Drops any box where patch covers >= 70% of the box's area."""
    result = []
    for b in boxes:
        orig_area = b.w * b.h
        if orig_area == 0:
            continue
        inter = _intersection_area(b, rx1, ry1, rx2, ry2)
        if inter / orig_area < 0.7:
            result.append(b)
    return result


def transform_bboxes_cutmix_b(
    boxes: List[BBox],
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> List[BBox]:
    """Map boxes from image B into the pasted region's coordinate space."""
    rw = rx2 - rx1
    rh = ry2 - ry1
    result = []
    for b in boxes:
        new_x = rx1 + b.x * rw
        new_y = ry1 + b.y * rh
        new_w = b.w * rw
        new_h = b.h * rh
        result.append(BBox(b.cls, new_x, new_y, new_w, new_h))
    return result
