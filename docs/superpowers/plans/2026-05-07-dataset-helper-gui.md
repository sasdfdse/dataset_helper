# Dataset Helper GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python tkinter GUI integrating video-to-frames extraction and YOLO dataset augmentation, output-compatible with Roboflow's dataset structure.

**Architecture:** Two-tab tkinter app (ttk.Notebook). `core/` holds all business logic — augmentation, label transform, video ops, batch pipeline. `gui/` holds only UI code and delegates all work to `core/`. Long-running tasks execute in background threads; progress is pushed to GUI via `root.after()`. No lambda functions anywhere — callbacks are always named methods.

**Tech Stack:** Python 3.10+, tkinter/ttk, opencv-python, numpy, Pillow, ffmpeg-python

---

## File Map

```
dataset_helper/augmentor/
├── main.py                  ← App entry point
├── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_label.py
│   ├── test_augment.py
│   └── test_pipeline.py
├── core/
│   ├── __init__.py
│   ├── label.py             ← BBox dataclass, YOLO read/write, bbox transforms
│   ├── augment.py           ← brightness, blur, noise, scale, cutmix functions
│   ├── pipeline.py          ← AugConfig, batch orchestration, file naming
│   └── video.py             ← ffmpeg-python: info, convert, extract frames
└── gui/
    ├── __init__.py
    ├── app.py               ← Main window, ttk.Notebook, tab wiring
    ├── augment_tab.py       ← Augmentation tab: SettingsPanel + PreviewPanel
    └── video_tab.py         ← Video Tool tab UI
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `augmentor/requirements.txt`
- Create: `augmentor/main.py`
- Create: `augmentor/core/__init__.py`
- Create: `augmentor/gui/__init__.py`
- Create: `augmentor/tests/__init__.py`

- [ ] **Step 1: Create the directory structure**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
mkdir -p augmentor/core augmentor/gui augmentor/tests
touch augmentor/core/__init__.py augmentor/gui/__init__.py augmentor/tests/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

`augmentor/requirements.txt`:
```
opencv-python
numpy
Pillow
ffmpeg-python
```

- [ ] **Step 3: Write main.py shell**

`augmentor/main.py`:
```python
from gui.app import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Install dependencies**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/
git commit -m "feat: scaffold augmentor package structure"
```

---

## Task 2: core/label.py

**Files:**
- Create: `augmentor/core/label.py`
- Create: `augmentor/tests/test_label.py`

- [ ] **Step 1: Write the failing tests**

`augmentor/tests/test_label.py`:
```python
import pytest
from pathlib import Path
from core.label import (
    BBox, read_labels, write_labels,
    clip_bbox, transform_bboxes_scale,
    filter_bboxes_cutmix_a, transform_bboxes_cutmix_b,
)


def test_read_labels_missing_file():
    assert read_labels(Path("/nonexistent/path.txt")) == []


def test_read_labels_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert read_labels(p) == []


def test_read_write_roundtrip(tmp_path):
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.4), BBox(1, 0.2, 0.3, 0.1, 0.1)]
    p = tmp_path / "test.txt"
    write_labels(p, boxes)
    loaded = read_labels(p)
    assert len(loaded) == 2
    assert loaded[0].cls == 0
    assert abs(loaded[0].x - 0.5) < 1e-5
    assert abs(loaded[1].w - 0.1) < 1e-5


def test_clip_bbox_fully_inside():
    b = BBox(0, 0.5, 0.5, 0.2, 0.2)
    result = clip_bbox(b, 0.0, 0.0, 1.0, 1.0)
    assert result is not None
    assert abs(result.x - 0.5) < 1e-5


def test_clip_bbox_fully_outside():
    b = BBox(0, 0.5, 0.5, 0.2, 0.2)
    result = clip_bbox(b, 0.8, 0.8, 1.0, 1.0)
    assert result is None


def test_clip_bbox_less_than_30_percent_remains():
    # box x=[0.0, 0.2], clip region x=[0.18, 1.0] → only 10% of width remains
    b = BBox(0, 0.1, 0.5, 0.2, 0.2)
    result = clip_bbox(b, 0.18, 0.0, 1.0, 1.0)
    assert result is None


def test_clip_bbox_partial_clip():
    # box x=[0.3, 0.7], clip region x=[0.0, 0.6]
    b = BBox(0, 0.5, 0.5, 0.4, 0.4)
    result = clip_bbox(b, 0.0, 0.0, 0.6, 1.0)
    assert result is not None
    assert abs(result.x - 0.45) < 1e-5
    assert abs(result.w - 0.3) < 1e-5


def test_transform_scale_up_center_box_unchanged():
    # Center box stays centered after zoom-in
    boxes = [BBox(0, 0.5, 0.5, 0.2, 0.2)]
    result = transform_bboxes_scale(boxes, 1.5)
    # offset = (1.5-1)/2 = 0.25, new_x = 0.5*1.5 - 0.25 = 0.5
    assert len(result) == 1
    assert abs(result[0].x - 0.5) < 1e-5
    assert abs(result[0].w - 0.3) < 1e-5


def test_transform_scale_up_edge_box_removed():
    # Box at far left edge goes out of frame on zoom-in
    boxes = [BBox(0, 0.02, 0.5, 0.04, 0.2)]
    result = transform_bboxes_scale(boxes, 2.0)
    assert len(result) == 0


def test_transform_scale_down_center_box_shrinks():
    boxes = [BBox(0, 0.5, 0.5, 0.2, 0.2)]
    result = transform_bboxes_scale(boxes, 0.5)
    # offset = (1-0.5)/2 = 0.25, new_x = 0.25 + 0.5*0.5 = 0.5, new_w = 0.2*0.5 = 0.1
    assert len(result) == 1
    assert abs(result[0].x - 0.5) < 1e-5
    assert abs(result[0].w - 0.1) < 1e-5


def test_filter_bboxes_cutmix_a_keeps_non_overlapping():
    b = BBox(0, 0.1, 0.1, 0.1, 0.1)
    result = filter_bboxes_cutmix_a([b], 0.5, 0.5, 0.9, 0.9)
    assert len(result) == 1


def test_filter_bboxes_cutmix_a_removes_mostly_covered():
    # Box fully inside patch region
    b = BBox(0, 0.7, 0.7, 0.2, 0.2)
    result = filter_bboxes_cutmix_a([b], 0.5, 0.5, 0.9, 0.9)
    assert len(result) == 0


def test_transform_bboxes_cutmix_b():
    # Box at center of B (0.5, 0.5, 0.4, 0.4) pasted into region (0.2, 0.2, 0.6, 0.6)
    b = BBox(0, 0.5, 0.5, 0.4, 0.4)
    result = transform_bboxes_cutmix_b([b], 0.2, 0.2, 0.6, 0.6)
    # region w=0.4, h=0.4
    # new_x = 0.2 + 0.5*0.4 = 0.4, new_w = 0.4*0.4 = 0.16
    assert len(result) == 1
    assert abs(result[0].x - 0.4) < 1e-5
    assert abs(result[0].w - 0.16) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/test_label.py -v
```

Expected: `ImportError: cannot import name 'BBox' from 'core.label'`

- [ ] **Step 3: Implement core/label.py**

`augmentor/core/label.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/test_label.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/
git commit -m "feat: add core/label.py with YOLO bbox read/write/transform"
```

---

## Task 3: core/augment.py

**Files:**
- Create: `augmentor/core/augment.py`
- Create: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Write the failing tests**

`augmentor/tests/test_augment.py`:
```python
import numpy as np
import pytest
from core.label import BBox
from core.augment import apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix


def _solid_img(value: int = 128, h: int = 100, w: int = 100) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# --- brightness ---

def test_brightness_increase():
    img = _solid_img(100)
    result = apply_brightness(img, 50)
    assert result.dtype == np.uint8
    assert np.all(result == 150)


def test_brightness_clip_high():
    img = _solid_img(200)
    result = apply_brightness(img, 100)
    assert np.all(result == 255)


def test_brightness_clip_low():
    img = _solid_img(50)
    result = apply_brightness(img, -100)
    assert np.all(result == 0)


# --- blur ---

def test_blur_preserves_shape():
    img = _solid_img()
    result = apply_blur(img, 3)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_blur_even_radius_accepted():
    img = _solid_img()
    result = apply_blur(img, 4)  # internally promoted to 5
    assert result.shape == img.shape


# --- noise ---

def test_noise_preserves_shape():
    img = _solid_img()
    result = apply_noise(img, 25.0)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_noise_values_in_range():
    img = _solid_img(128)
    result = apply_noise(img, 80.0)
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


# --- scale ---

def test_scale_up_preserves_shape():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    out_img, out_boxes = apply_scale(img, boxes, 1.5)
    assert out_img.shape == img.shape


def test_scale_down_preserves_shape():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    out_img, out_boxes = apply_scale(img, boxes, 0.7)
    assert out_img.shape == img.shape


def test_scale_down_corners_are_black():
    img = _solid_img(200)
    boxes = []
    out_img, _ = apply_scale(img, boxes, 0.4)
    assert int(out_img[0, 0, 0]) == 0   # top-left corner is padding
    assert int(out_img[99, 99, 0]) == 0  # bottom-right corner is padding


def test_scale_up_center_still_bright():
    img = _solid_img(200)
    boxes = []
    out_img, _ = apply_scale(img, boxes, 1.5)
    assert int(out_img[50, 50, 0]) == 200  # center should remain bright


# --- cutmix ---

def test_cutmix_preserves_shape():
    img_a = _solid_img(100)
    img_b = _solid_img(200)
    boxes_a = [BBox(0, 0.1, 0.1, 0.1, 0.1)]
    boxes_b = [BBox(0, 0.5, 0.5, 0.2, 0.2)]
    out_img, out_boxes = apply_cutmix(img_a, boxes_a, img_b, boxes_b)
    assert out_img.shape == img_a.shape
    assert isinstance(out_boxes, list)


def test_cutmix_patch_region_differs():
    img_a = _solid_img(0)
    img_b = _solid_img(255)
    out_img, _ = apply_cutmix(img_a, [], img_b, [])
    # Some pixels should be 255 (from B) and some 0 (from A)
    assert int(out_img.max()) == 255
    assert int(out_img.min()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/test_augment.py -v
```

Expected: `ImportError: cannot import name 'apply_brightness' from 'core.augment'`

- [ ] **Step 3: Implement core/augment.py**

`augmentor/core/augment.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/test_augment.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/
git commit -m "feat: add core/augment.py with all five augmentation functions"
```

---

## Task 4: core/pipeline.py

**Files:**
- Create: `augmentor/core/pipeline.py`
- Create: `augmentor/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

`augmentor/tests/test_pipeline.py`:
```python
import cv2
import numpy as np
import pytest
from pathlib import Path
from core.pipeline import AugConfig, run_batch, _collect_originals


def _make_dataset(tmp_path: Path, n: int = 3):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(n):
        img = np.full((100, 100, 3), i * 60 + 30, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i:03d}.jpg"), img)
        (labels_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return images_dir, labels_dir


def _noop(done: int, total: int) -> None:
    pass


def test_collect_originals_excludes_augmented(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "real.jpg").touch()
    (images_dir / "real_aug1.jpg").touch()
    (images_dir / "a_b_cutmix_1.jpg").touch()
    result = _collect_originals(images_dir)
    assert len(result) == 1
    assert result[0].name == "real.jpg"


def test_run_batch_multiplier_creates_correct_count(tmp_path):
    images_dir, labels_dir = _make_dataset(tmp_path, n=3)
    config = AugConfig(brightness_enabled=True, multiplier=2)
    run_batch(images_dir, labels_dir, config, progress_cb=_noop)
    aug_imgs = list(images_dir.glob("*_aug*.jpg"))
    aug_lbls = list(labels_dir.glob("*_aug*.txt"))
    assert len(aug_imgs) == 6   # 3 originals * 2
    assert len(aug_lbls) == 6


def test_run_batch_image_label_names_match(tmp_path):
    images_dir, labels_dir = _make_dataset(tmp_path, n=2)
    config = AugConfig(brightness_enabled=True, multiplier=1)
    run_batch(images_dir, labels_dir, config, progress_cb=_noop)
    for img_path in images_dir.glob("*_aug*.jpg"):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        assert lbl_path.exists(), f"Missing label for {img_path.name}"


def test_run_batch_progress_called(tmp_path):
    images_dir, labels_dir = _make_dataset(tmp_path, n=2)
    config = AugConfig(brightness_enabled=True, multiplier=3)
    calls = []

    def record(done: int, total: int) -> None:
        calls.append((done, total))

    run_batch(images_dir, labels_dir, config, progress_cb=record)
    assert len(calls) == 6
    assert calls[-1][0] == calls[-1][1]  # final call: done == total


def test_run_batch_cutmix_creates_files(tmp_path):
    images_dir, labels_dir = _make_dataset(tmp_path, n=3)
    config = AugConfig(cutmix_enabled=True, cutmix_pairs=2, multiplier=0)
    run_batch(images_dir, labels_dir, config, progress_cb=_noop)
    cutmix_imgs = list(images_dir.glob("*_cutmix_*.jpg"))
    assert len(cutmix_imgs) == 2


def test_run_batch_no_augmentations_no_output(tmp_path):
    images_dir, labels_dir = _make_dataset(tmp_path, n=2)
    config = AugConfig(multiplier=2)  # all disabled
    run_batch(images_dir, labels_dir, config, progress_cb=_noop)
    # With no augmentations enabled, _apply_augmentations returns original unchanged
    aug_imgs = list(images_dir.glob("*_aug*.jpg"))
    assert len(aug_imgs) == 4  # files are still created, just unmodified
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/test_pipeline.py -v
```

Expected: `ImportError: cannot import name 'AugConfig' from 'core.pipeline'`

- [ ] **Step 3: Implement core/pipeline.py**

`augmentor/core/pipeline.py`:
```python
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Callable, Tuple

import cv2
import numpy as np

from core.label import BBox, read_labels, write_labels
from core.augment import apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix


@dataclass
class AugConfig:
    brightness_enabled: bool = False
    brightness_min: int = -50
    brightness_max: int = 50
    blur_enabled: bool = False
    blur_radius: int = 3
    noise_enabled: bool = False
    noise_strength: float = 25.0
    scale_enabled: bool = False
    scale_min: float = 0.8
    scale_max: float = 1.2
    cutmix_enabled: bool = False
    cutmix_pairs: int = 10
    multiplier: int = 3


def _collect_originals(images_dir: Path) -> List[Path]:
    all_imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    result = []
    for p in all_imgs:
        if "_aug" not in p.stem and "_cutmix" not in p.stem:
            result.append(p)
    return sorted(result)


def _apply_augmentations(
    img: np.ndarray,
    boxes: List[BBox],
    config: AugConfig,
) -> Tuple[np.ndarray, List[BBox]]:
    enabled = []
    if config.brightness_enabled:
        enabled.append("brightness")
    if config.blur_enabled:
        enabled.append("blur")
    if config.noise_enabled:
        enabled.append("noise")
    if config.scale_enabled:
        enabled.append("scale")

    if not enabled:
        return img, boxes

    count = random.randint(1, len(enabled))
    chosen = random.sample(enabled, k=count)

    for aug in chosen:
        if aug == "brightness":
            delta = random.randint(config.brightness_min, config.brightness_max)
            img = apply_brightness(img, delta)
        elif aug == "blur":
            img = apply_blur(img, config.blur_radius)
        elif aug == "noise":
            img = apply_noise(img, config.noise_strength)
        elif aug == "scale":
            scale = random.uniform(config.scale_min, config.scale_max)
            img, boxes = apply_scale(img, boxes, scale)

    return img, boxes


def run_batch(
    images_dir: Path,
    labels_dir: Path,
    config: AugConfig,
    progress_cb: Callable[[int, int], None],
) -> None:
    originals = _collect_originals(images_dir)
    cutmix_count = config.cutmix_pairs if config.cutmix_enabled else 0
    total = len(originals) * config.multiplier + cutmix_count
    done = 0

    for img_path in originals:
        label_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = read_labels(label_path)

        for i in range(config.multiplier):
            aug_img, aug_boxes = _apply_augmentations(img.copy(), list(boxes), config)
            out_stem = f"{img_path.stem}_aug{i + 1}"
            cv2.imwrite(str(images_dir / (out_stem + img_path.suffix)), aug_img)
            write_labels(labels_dir / (out_stem + ".txt"), aug_boxes)
            done += 1
            progress_cb(done, total)

    if config.cutmix_enabled and len(originals) >= 2:
        for i in range(config.cutmix_pairs):
            img_a_path, img_b_path = random.sample(originals, 2)
            img_a = cv2.imread(str(img_a_path))
            img_b = cv2.imread(str(img_b_path))
            boxes_a = read_labels(labels_dir / (img_a_path.stem + ".txt"))
            boxes_b = read_labels(labels_dir / (img_b_path.stem + ".txt"))

            mixed_img, mixed_boxes = apply_cutmix(img_a, boxes_a, img_b, boxes_b)
            out_stem = f"{img_a_path.stem}_{img_b_path.stem}_cutmix_{i + 1}"
            cv2.imwrite(str(images_dir / (out_stem + img_a_path.suffix)), mixed_img)
            write_labels(labels_dir / (out_stem + ".txt"), mixed_boxes)
            done += 1
            progress_cb(done, total)
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/ -v
```

Expected: all tests PASS (including label and augment from previous tasks).

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/
git commit -m "feat: add core/pipeline.py with AugConfig and batch augmentation runner"
```

---

## Task 5: core/video.py

**Files:**
- Create: `augmentor/core/video.py`

No unit tests — requires actual video files. Verify manually at integration time.

- [ ] **Step 1: Implement core/video.py**

`augmentor/core/video.py`:
```python
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import cv2
import ffmpeg


@dataclass
class VideoInfo:
    format: str = ""
    duration_sec: float = 0.0
    bitrate_bps: int = 0
    video_codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    nb_frames: int = 0
    audio_codec: str = ""
    sample_rate: str = ""
    channels: int = 0
    error: str = ""

    def is_valid(self) -> bool:
        return self.error == ""


def get_video_info(video_path: Path) -> VideoInfo:
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as e:
        return VideoInfo(error=str(e))

    info = VideoInfo()
    fmt = probe.get("format", {})
    info.format = fmt.get("format_long_name", "")
    info.duration_sec = float(fmt.get("duration", 0))
    info.bitrate_bps = int(fmt.get("bit_rate", 0))

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            info.video_codec = stream.get("codec_name", "")
            info.width = stream.get("width", 0)
            info.height = stream.get("height", 0)
            r = stream.get("r_frame_rate", "0/1").split("/")
            if len(r) == 2 and int(r[1]) != 0:
                info.fps = int(r[0]) / int(r[1])
            raw_frames = stream.get("nb_frames", "")
            if raw_frames.isdigit():
                info.nb_frames = int(raw_frames)
        elif stream.get("codec_type") == "audio":
            info.audio_codec = stream.get("codec_name", "")
            info.sample_rate = stream.get("sample_rate", "")
            info.channels = stream.get("channels", 0)

    return info


def convert_to_mp4(
    input_path: Path,
    output_path: Path,
    progress_cb: Callable[[int], None],
) -> None:
    """Convert any video to MP4 (H.264 + AAC). Raises RuntimeError on failure."""
    info = get_video_info(input_path)
    total_duration = info.duration_sec

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-progress", "pipe:1",
        "-nostats",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                if total_duration > 0:
                    pct = min(int(ms / 1_000_000 / total_duration * 100), 99)
                    progress_cb(pct)
            except (ValueError, IndexError):
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    progress_cb(100)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    ratio: int,
    grayscale: bool,
    progress_cb: Callable[[int, int], None],
) -> int:
    """Extract frames from video. ratio=1 → all, 2 → every other, 3 → every third.
    Returns number of saved frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % ratio == 0:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1
        if total > 0:
            progress_cb(frame_idx, total)

    cap.release()
    return saved
```

- [ ] **Step 2: Smoke-test imports**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -c "from core.video import get_video_info, convert_to_mp4, extract_frames; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/core/video.py
git commit -m "feat: add core/video.py with ffmpeg convert and frame extract"
```

---

## Task 6: gui/app.py — Main Window

**Files:**
- Create: `augmentor/gui/app.py`

- [ ] **Step 1: Implement gui/app.py**

`augmentor/gui/app.py`:
```python
import tkinter as tk
from tkinter import ttk

from gui.video_tab import VideoTab
from gui.augment_tab import AugmentTab


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Helper — ROBIT")
        self.geometry("1150x720")
        self.minsize(900, 600)
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        video_frame = ttk.Frame(notebook)
        aug_frame = ttk.Frame(notebook)

        notebook.add(video_frame, text="  📹 Video Tool  ")
        notebook.add(aug_frame, text="  🔧 Augmentation  ")

        VideoTab(video_frame).pack(fill=tk.BOTH, expand=True)
        AugmentTab(aug_frame).pack(fill=tk.BOTH, expand=True)
```

- [ ] **Step 2: Smoke-test (app won't fully open yet — tabs not implemented)**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -c "from gui.app import App; print('import OK')"
```

Expected: `ImportError` for missing VideoTab/AugmentTab — that's expected until next tasks.

- [ ] **Step 3: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/gui/app.py
git commit -m "feat: add gui/app.py main window with tabbed layout"
```

---

## Task 7: gui/augment_tab.py

**Files:**
- Create: `augmentor/gui/augment_tab.py`

- [ ] **Step 1: Implement gui/augment_tab.py**

`augmentor/gui/augment_tab.py`:
```python
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

from core.label import BBox, read_labels
from core.pipeline import AugConfig, run_batch, _collect_originals


_PREVIEW_SIZE = (380, 280)
_BBOX_COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44",
    "#FF44FF", "#44FFFF", "#FF8844", "#44FF88",
]


class AugmentTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._preview_pairs: List[Tuple[Path, Path, Path, Path]] = []
        self._preview_idx: int = 0
        self._running: bool = False
        self._build_ui()

    # ── Layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=240)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_settings(self)
        self._build_preview(self)

    def _build_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="Settings", padding=8)
        frame.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)

        # Folder selection
        ttk.Label(frame, text="images/ folder:").grid(row=0, column=0, sticky="w")
        self._images_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._images_var, width=22).grid(row=1, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_images).grid(row=1, column=1, padx=(4, 0))

        ttk.Label(frame, text="labels/ folder:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._labels_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._labels_var, width=22).grid(row=3, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_labels).grid(row=3, column=1, padx=(4, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)

        # Augmentation controls
        row = 5
        row = self._aug_section(frame, row, "Brightness", "brightness",
                                [("min", -100, 100, -50), ("max", -100, 100, 50)])
        row = self._aug_section(frame, row, "Blur", "blur",
                                [("radius", 1, 15, 3)])
        row = self._aug_section(frame, row, "Noise", "noise",
                                [("strength", 0, 100, 25)])
        row = self._aug_section(frame, row, "Scale", "scale",
                                [("min ×", 50, 200, 80), ("max ×", 50, 200, 120)])
        row = self._aug_section(frame, row, "CutMix", "cutmix",
                                [("pairs", 1, 100, 10)])

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(frame, text="Multiplier (×):").grid(row=row, column=0, sticky="w")
        self._multiplier_var = tk.IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=50, textvariable=self._multiplier_var, width=6).grid(
            row=row, column=1, sticky="w")

        frame.columnconfigure(0, weight=1)

    def _aug_section(self, parent, row: int, label: str, key: str, params) -> int:
        enabled_var = tk.BooleanVar(value=False)
        setattr(self, f"_{key}_enabled", enabled_var)
        ttk.Checkbutton(parent, text=label, variable=enabled_var).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for name, lo, hi, default in params:
            ttk.Label(parent, text=f"  {name}:").grid(row=row, column=0, sticky="w")
            var = tk.IntVar(value=default)
            setattr(self, f"_{key}_{name.replace(' ', '_').replace('×', 'x')}_var", var)
            ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal").grid(
                row=row, column=1, sticky="ew")
            row += 1
        return row

    def _build_preview(self, parent):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)

        # Image preview area
        img_frame = ttk.LabelFrame(frame, text="Preview", padding=4)
        img_frame.grid(row=0, column=0, sticky="nsew")
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)

        ttk.Label(img_frame, text="Original").grid(row=0, column=0)
        ttk.Label(img_frame, text="Augmented").grid(row=0, column=1)

        self._orig_lbl = ttk.Label(img_frame, background="#1e1e1e")
        self._orig_lbl.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")

        self._aug_lbl = ttk.Label(img_frame, background="#1e1e1e")
        self._aug_lbl.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")

        nav_frame = ttk.Frame(img_frame)
        nav_frame.grid(row=2, column=0, columnspan=2)
        ttk.Button(nav_frame, text="◀ Prev", command=self._prev_preview).pack(side=tk.LEFT, padx=8)
        self._preview_lbl = ttk.Label(nav_frame, text="–")
        self._preview_lbl.pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next ▶", command=self._next_preview).pack(side=tk.LEFT, padx=8)

        # Progress + run
        bottom = ttk.Frame(frame)
        bottom.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(bottom, variable=self._progress_var, maximum=100)
        self._progress_bar.grid(row=0, column=0, sticky="ew", padx=4)

        self._status_lbl = ttk.Label(bottom, text="Ready")
        self._status_lbl.grid(row=1, column=0)

        self._run_btn = ttk.Button(bottom, text="▶ Run Augmentation", command=self._on_run)
        self._run_btn.grid(row=2, column=0, pady=6)

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_images(self):
        path = filedialog.askdirectory(title="Select images/ folder")
        if path:
            self._images_var.set(path)
            self._images_dir = Path(path)

    def _browse_labels(self):
        path = filedialog.askdirectory(title="Select labels/ folder")
        if path:
            self._labels_var.set(path)
            self._labels_dir = Path(path)

    # ── Run ─────────────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        if self._images_dir is None or self._labels_dir is None:
            messagebox.showerror("Error", "Select both images/ and labels/ folders first.")
            return

        config = self._build_config()
        self._running = True
        self._run_btn.configure(state=tk.DISABLED)
        self._progress_var.set(0)
        self._status_lbl.configure(text="Running…")

        thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        thread.start()

    def _build_config(self) -> AugConfig:
        return AugConfig(
            brightness_enabled=self._brightness_enabled.get(),
            brightness_min=self._brightness_min_var.get(),
            brightness_max=self._brightness_max_var.get(),
            blur_enabled=self._blur_enabled.get(),
            blur_radius=self._blur_radius_var.get(),
            noise_enabled=self._noise_enabled.get(),
            noise_strength=float(self._noise_strength_var.get()),
            scale_enabled=self._scale_enabled.get(),
            scale_min=self._scale_min_x_var.get() / 100.0,
            scale_max=self._scale_max_x_var.get() / 100.0,
            cutmix_enabled=self._cutmix_enabled.get(),
            cutmix_pairs=self._cutmix_pairs_var.get(),
            multiplier=self._multiplier_var.get(),
        )

    def _run_worker(self, config: AugConfig):
        try:
            run_batch(
                self._images_dir,
                self._labels_dir,
                config,
                progress_cb=self._on_progress,
            )
            self.after(0, self._on_complete)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _on_progress(self, done: int, total: int):
        pct = (done / total * 100) if total > 0 else 100
        self.after(0, self._update_progress, pct, done, total)

    def _update_progress(self, pct: float, done: int, total: int):
        self._progress_var.set(pct)
        self._status_lbl.configure(text=f"{done} / {total}")

    def _on_complete(self):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="Done!")
        self._load_preview_list()

    def _on_error(self, msg: str):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="Error")
        messagebox.showerror("Augmentation Error", msg)

    # ── Preview ──────────────────────────────────────────────────────

    def _load_preview_list(self):
        if self._images_dir is None:
            return
        aug_imgs = sorted(
            p for p in self._images_dir.glob("*.jpg")
            if "_aug" in p.stem or "_cutmix" in p.stem
        )
        self._preview_pairs = []
        for aug_path in aug_imgs:
            aug_label = self._labels_dir / (aug_path.stem + ".txt")
            # Find original stem
            if "_cutmix" in aug_path.stem:
                orig_path = aug_path
                orig_label = aug_label
            else:
                orig_stem = aug_path.stem.rsplit("_aug", 1)[0]
                orig_path = self._images_dir / (orig_stem + aug_path.suffix)
                orig_label = self._labels_dir / (orig_stem + ".txt")
            self._preview_pairs.append((orig_path, orig_label, aug_path, aug_label))

        self._preview_idx = 0
        if self._preview_pairs:
            self._show_preview(0)

    def _show_preview(self, idx: int):
        orig_img_path, orig_lbl_path, aug_img_path, aug_lbl_path = self._preview_pairs[idx]
        orig_img = cv2.imread(str(orig_img_path))
        aug_img = cv2.imread(str(aug_img_path))
        orig_boxes = read_labels(orig_lbl_path)
        aug_boxes = read_labels(aug_lbl_path)

        orig_photo = self._to_photoimage(self._draw_bboxes(orig_img, orig_boxes))
        aug_photo = self._to_photoimage(self._draw_bboxes(aug_img, aug_boxes))

        self._orig_lbl.configure(image=orig_photo)
        self._orig_lbl.image = orig_photo
        self._aug_lbl.configure(image=aug_photo)
        self._aug_lbl.image = aug_photo

        total = len(self._preview_pairs)
        self._preview_lbl.configure(text=f"{idx + 1} / {total}")

    def _draw_bboxes(self, img: np.ndarray, boxes: List[BBox]) -> np.ndarray:
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        result = img.copy()
        h, w = result.shape[:2]
        for i, b in enumerate(boxes):
            color_hex = _BBOX_COLORS[i % len(_BBOX_COLORS)]
            color_bgr = tuple(int(color_hex[j:j+2], 16) for j in (5, 3, 1))
            x1 = int((b.x - b.w / 2) * w)
            y1 = int((b.y - b.h / 2) * h)
            x2 = int((b.x + b.w / 2) * w)
            y2 = int((b.y + b.h / 2) * h)
            cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 2)
        return result

    def _to_photoimage(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize(_PREVIEW_SIZE, Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _prev_preview(self):
        if not self._preview_pairs:
            return
        self._preview_idx = (self._preview_idx - 1) % len(self._preview_pairs)
        self._show_preview(self._preview_idx)

    def _next_preview(self):
        if not self._preview_pairs:
            return
        self._preview_idx = (self._preview_idx + 1) % len(self._preview_pairs)
        self._show_preview(self._preview_idx)
```

- [ ] **Step 2: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/gui/augment_tab.py
git commit -m "feat: add gui/augment_tab.py with settings, preview, and run integration"
```

---

## Task 8: gui/video_tab.py

**Files:**
- Create: `augmentor/gui/video_tab.py`

- [ ] **Step 1: Implement gui/video_tab.py**

`augmentor/gui/video_tab.py`:
```python
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.video import VideoInfo, get_video_info, convert_to_mp4, extract_frames


_MODES = ["WEBM → MP4 변환", "동영상 → 프레임 추출", "WEBM → MP4 → 프레임 추출 (연속)", "동영상 정보 보기"]


class VideoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._running = False
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)

        # Input
        inp_frame = ttk.LabelFrame(self, text="Input", padding=8)
        inp_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        inp_frame.columnconfigure(1, weight=1)

        ttk.Label(inp_frame, text="입력 파일:").grid(row=0, column=0, sticky="w")
        self._input_var = tk.StringVar()
        ttk.Entry(inp_frame, textvariable=self._input_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(inp_frame, text="Browse", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(inp_frame, text="출력 경로:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._output_var = tk.StringVar()
        ttk.Entry(inp_frame, textvariable=self._output_var).grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(inp_frame, text="Browse", command=self._browse_output).grid(row=1, column=2, pady=(4, 0))

        # Mode selection
        mode_frame = ttk.LabelFrame(self, text="작업 선택", padding=8)
        mode_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._mode_var = tk.IntVar(value=0)
        for i, label in enumerate(_MODES):
            rb = ttk.Radiobutton(mode_frame, text=label, variable=self._mode_var, value=i,
                                 command=self._on_mode_change)
            rb.pack(anchor="w")

        # Extract options
        self._extract_frame = ttk.LabelFrame(self, text="프레임 추출 옵션", padding=8)
        self._extract_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)

        ttk.Label(self._extract_frame, text="추출 비율:").pack(side=tk.LEFT)
        self._ratio_var = tk.IntVar(value=1)
        for val, label in [(1, "전체"), (2, "1/2"), (3, "1/3")]:
            ttk.Radiobutton(self._extract_frame, text=label, variable=self._ratio_var, value=val).pack(
                side=tk.LEFT, padx=4)

        self._gray_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._extract_frame, text="그레이스케일", variable=self._gray_var).pack(
            side=tk.LEFT, padx=8)

        # Info display
        self._info_frame = ttk.LabelFrame(self, text="동영상 정보", padding=8)
        self._info_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self._info_text = tk.Text(self._info_frame, height=7, state=tk.DISABLED, font=("Courier", 10))
        self._info_text.pack(fill=tk.X)

        # Progress + run
        bottom = ttk.Frame(self)
        bottom.grid(row=4, column=0, sticky="ew", padx=8, pady=8)
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(bottom, variable=self._progress_var, maximum=100).grid(
            row=0, column=0, sticky="ew")

        self._status_lbl = ttk.Label(bottom, text="Ready")
        self._status_lbl.grid(row=1, column=0)

        self._run_btn = ttk.Button(bottom, text="▶ Run", command=self._on_run)
        self._run_btn.grid(row=2, column=0, pady=4)

        self._on_mode_change()  # set initial visibility

    # ── File browsing ────────────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.webm *.mp4 *.avi *.mkv *.mov"), ("All", "*.*")],
        )
        if not path:
            return
        self._input_var.set(path)
        inp = Path(path)
        mode = self._mode_var.get()
        if mode == 0:
            self._output_var.set(str(inp.with_suffix(".mp4")))
        elif mode in (1, 2):
            self._output_var.set(str(inp.parent / (inp.stem + "_frames")))

    def _browse_output(self):
        mode = self._mode_var.get()
        if mode in (1, 2):
            path = filedialog.askdirectory(title="Select output directory")
        else:
            path = filedialog.asksaveasfilename(
                title="Save MP4 as",
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4")],
            )
        if path:
            self._output_var.set(path)

    # ── Mode switch ──────────────────────────────────────────────────

    def _on_mode_change(self):
        mode = self._mode_var.get()
        if mode in (1, 2):
            self._extract_frame.grid()
        else:
            self._extract_frame.grid_remove()
        if mode == 3:
            self._info_frame.grid()
        else:
            self._info_frame.grid_remove()

    # ── Run ─────────────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        mode = self._mode_var.get()
        inp = self._input_var.get().strip()
        if not inp:
            messagebox.showerror("Error", "입력 파일을 선택하세요.")
            return

        self._running = True
        self._run_btn.configure(state=tk.DISABLED)
        self._progress_var.set(0)
        self._status_lbl.configure(text="Running…")

        thread = threading.Thread(target=self._run_worker, args=(mode, inp), daemon=True)
        thread.start()

    def _run_worker(self, mode: int, inp: str):
        try:
            inp_path = Path(inp)
            out = self._output_var.get().strip()

            if mode == 0:
                self._do_convert(inp_path, Path(out))
            elif mode == 1:
                self._do_extract(inp_path, Path(out))
            elif mode == 2:
                mp4_path = inp_path.with_suffix(".mp4")
                self._do_convert(inp_path, mp4_path)
                self._do_extract(mp4_path, Path(out))
            elif mode == 3:
                self._do_info(inp_path)

            self.after(0, self._on_complete)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _do_convert(self, inp: Path, out: Path):
        self.after(0, self._status_lbl.configure, {"text": "변환 중…"})
        convert_to_mp4(inp, out, progress_cb=self._on_convert_progress)

    def _do_extract(self, inp: Path, out: Path):
        self.after(0, self._status_lbl.configure, {"text": "프레임 추출 중…"})
        saved = extract_frames(
            inp, out,
            ratio=self._ratio_var.get(),
            grayscale=self._gray_var.get(),
            progress_cb=self._on_extract_progress,
        )
        self.after(0, self._status_lbl.configure, {"text": f"추출 완료: {saved}장"})

    def _do_info(self, inp: Path):
        info = get_video_info(inp)
        self.after(0, self._display_info, info)

    def _display_info(self, info: VideoInfo):
        lines = []
        if not info.is_valid():
            lines.append(f"오류: {info.error}")
        else:
            lines.append(f"포맷:      {info.format}")
            mins = int(info.duration_sec) // 60
            secs = int(info.duration_sec) % 60
            lines.append(f"길이:      {mins}분 {secs}초")
            if info.bitrate_bps:
                lines.append(f"비트레이트: {info.bitrate_bps // 1000} kbps")
            if info.video_codec:
                lines.append(f"비디오:    {info.video_codec}  {info.width}×{info.height}  {info.fps:.2f} fps")
            if info.audio_codec:
                lines.append(f"오디오:    {info.audio_codec}  {info.sample_rate} Hz  {info.channels}ch")

        self._info_text.configure(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, "\n".join(lines))
        self._info_text.configure(state=tk.DISABLED)

    def _on_convert_progress(self, pct: int):
        self.after(0, self._progress_var.set, float(pct))

    def _on_extract_progress(self, done: int, total: int):
        pct = (done / total * 100) if total > 0 else 100
        self.after(0, self._progress_var.set, pct)

    def _on_complete(self):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._progress_var.set(100)
        self._status_lbl.configure(text="완료!")

    def _on_error(self, msg: str):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="오류 발생")
        messagebox.showerror("오류", msg)
```

- [ ] **Step 2: Smoke-test the full app**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python main.py
```

Expected: GUI window opens with two tabs, no errors in terminal.

- [ ] **Step 3: Run the full test suite one last time**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper
git add augmentor/
git commit -m "feat: complete Dataset Helper GUI with Video Tool and Augmentation tabs"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Brightness augmentation | Task 3 `apply_brightness` |
| Blur augmentation | Task 3 `apply_blur` |
| Noise augmentation | Task 3 `apply_noise` |
| Scale augmentation + bbox transform | Task 3 `apply_scale`, Task 2 `transform_bboxes_scale` |
| CutMix + bbox handling | Task 3 `apply_cutmix`, Task 2 `filter/transform cutmix` |
| YOLO label read/write | Task 2 `read_labels`, `write_labels` |
| Roboflow folder structure | Task 4 `_collect_originals`, file naming in `run_batch` |
| Batch + preview (C option) | Task 4 `run_batch`, Task 7 preview panel |
| CutMix pairs count (C option) | Task 4 `AugConfig.cutmix_pairs` |
| Video WEBM→MP4 | Task 5 `convert_to_mp4` |
| Video frame extraction | Task 5 `extract_frames` |
| Video info display | Task 5 `get_video_info`, Task 8 `_display_info` |
| No lambda functions | All callbacks are named methods throughout |
| Progress bar + threading | Tasks 7, 8 — `threading.Thread` + `after()` |

**No placeholders found.**

**Type consistency confirmed:** `BBox`, `AugConfig`, `VideoInfo` used consistently across all tasks.
