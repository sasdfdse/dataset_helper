# Augmentation Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 10 new augmentation types (Flip H/V, Rotation, Crop, Shear, Grayscale, Saturation, Exposure, Motion Blur, Camera Gain, Mosaic) to the dataset_helper augmentor package.

**Architecture:** New `apply_*` functions added to `augment.py`; `AugConfig` and pipeline execution logic extended in `pipeline.py`; GUI settings panel made scrollable and extended with new toggle+slider sections in `augment_tab.py`. All functions follow the existing pattern: take `np.ndarray` (and optionally `List[BBox]`), return transformed image (and updated boxes).

**Tech Stack:** Python 3.10, OpenCV (`cv2`), NumPy, tkinter/ttk, pytest

---

## File Map

| File | Change |
|---|---|
| `augmentor/core/augment.py` | Add 10 new `apply_*` functions |
| `augmentor/core/pipeline.py` | Extend `AugConfig`; update `_collect_originals`, `_apply_augmentations`, `preview_augment`, `run_batch` |
| `augmentor/gui/augment_tab.py` | Make settings scrollable; add 11 new `_aug_section` calls; update `_build_config` |
| `augmentor/tests/test_augment.py` | Add tests for each new augmentation function |
| `augmentor/tests/test_pipeline.py` | Add test for mosaic pipeline integration |

---

### Task 1: Flip Augmentation

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests for `apply_flip`**

Append to `augmentor/tests/test_augment.py`:

```python
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip,
)


def test_flip_horizontal_updates_x():
    img = _solid_img()
    boxes = [BBox(0, 0.2, 0.5, 0.1, 0.1)]
    out_img, out_boxes = apply_flip(img, boxes, horizontal=True, vertical=False)
    assert out_img.shape == img.shape
    assert abs(out_boxes[0].x - 0.8) < 1e-6  # 1 - 0.2


def test_flip_vertical_updates_y():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.3, 0.1, 0.1)]
    out_img, out_boxes = apply_flip(img, boxes, horizontal=False, vertical=True)
    assert abs(out_boxes[0].y - 0.7) < 1e-6  # 1 - 0.3


def test_flip_both_updates_x_and_y():
    img = _solid_img()
    boxes = [BBox(0, 0.2, 0.3, 0.1, 0.1)]
    _, out_boxes = apply_flip(img, boxes, horizontal=True, vertical=True)
    assert abs(out_boxes[0].x - 0.8) < 1e-6
    assert abs(out_boxes[0].y - 0.7) < 1e-6


def test_flip_preserves_size_and_w_h():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.2)]
    _, out_boxes = apply_flip(img, boxes, horizontal=True, vertical=True)
    assert abs(out_boxes[0].w - 0.3) < 1e-6
    assert abs(out_boxes[0].h - 0.2) < 1e-6


def test_flip_horizontal_mirrors_pixels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, 0] = 255  # left column bright
    out_img, _ = apply_flip(img, [], horizontal=True, vertical=False)
    assert int(out_img[0, 9, 0]) == 255  # right column should now be bright
    assert int(out_img[0, 0, 0]) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_flip_horizontal_updates_x -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement `apply_flip` in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -k "flip" -v 2>&1 | tail -10
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add apply_flip augmentation with bbox support"
```

---

### Task 2: Rotation Augmentation

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests for `apply_rotation`**

Update the import line in `test_augment.py` to add `apply_rotation`, then append tests:

```python
# update import line at top of test_augment.py:
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation,
)


def test_rotation_preserves_shape():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    out_img, out_boxes = apply_rotation(img, boxes, 45)
    assert out_img.shape == img.shape


def test_rotation_zero_keeps_center_box():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    _, out_boxes = apply_rotation(img, boxes, 0)
    assert len(out_boxes) == 1
    assert abs(out_boxes[0].x - 0.5) < 0.01
    assert abs(out_boxes[0].y - 0.5) < 0.01


def test_rotation_90_swaps_xy_for_center_box():
    img = _solid_img(h=100, w=100)
    boxes = [BBox(0, 0.5, 0.5, 0.4, 0.2)]
    _, out_boxes = apply_rotation(img, boxes, 90)
    assert len(out_boxes) == 1
    # After 90° rotation of a square image, center box stays near center
    assert abs(out_boxes[0].x - 0.5) < 0.05
    assert abs(out_boxes[0].y - 0.5) < 0.05


def test_rotation_out_of_bounds_box_removed():
    img = _solid_img()
    # Box entirely in the top-left corner; large rotation clips it out
    boxes = [BBox(0, 0.02, 0.02, 0.02, 0.02)]
    _, out_boxes = apply_rotation(img, boxes, 45)
    # Box may or may not survive - just check no crash and shape is fine
    assert isinstance(out_boxes, list)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_rotation_preserves_shape -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement `apply_rotation` in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -k "rotation" -v 2>&1 | tail -10
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add apply_rotation augmentation with bbox support"
```

---

### Task 3: Crop Augmentation

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests for `apply_crop`**

Update import line to add `apply_crop`, then append:

```python
# update import line:
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop,
)


def test_crop_preserves_shape():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    out_img, out_boxes = apply_crop(img, boxes, 0.8)
    assert out_img.shape == img.shape


def test_crop_keeps_fully_inside_box():
    img = _solid_img()
    # With crop_ratio=1.0 nothing is cropped
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    _, out_boxes = apply_crop(img, boxes, 1.0)
    assert len(out_boxes) == 1


def test_crop_output_dtype():
    img = _solid_img()
    out_img, _ = apply_crop(img, [], 0.9)
    assert out_img.dtype == np.uint8
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_crop_preserves_shape -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement `apply_crop` in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -k "crop" -v 2>&1 | tail -10
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add apply_crop augmentation with bbox support"
```

---

### Task 4: Shear Augmentation

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests for `apply_shear`**

Update import line to add `apply_shear`, then append:

```python
# update import line:
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop, apply_shear,
)


def test_shear_preserves_shape():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    out_img, out_boxes = apply_shear(img, boxes, 10, 0)
    assert out_img.shape == img.shape


def test_shear_zero_keeps_center_box():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    _, out_boxes = apply_shear(img, boxes, 0, 0)
    assert len(out_boxes) == 1
    assert abs(out_boxes[0].x - 0.5) < 0.01
    assert abs(out_boxes[0].y - 0.5) < 0.01


def test_shear_output_dtype():
    img = _solid_img()
    out_img, _ = apply_shear(img, [], 5, 5)
    assert out_img.dtype == np.uint8
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_shear_preserves_shape -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement `apply_shear` in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
def apply_shear(
    img: np.ndarray,
    boxes: List[BBox],
    shear_x: float,
    shear_y: float,
) -> Tuple[np.ndarray, List[BBox]]:
    h, w = img.shape[:2]
    sx = np.tan(np.radians(shear_x))
    sy = np.tan(np.radians(shear_y))
    M = np.float32([
        [1, sx, -sx * h / 2],
        [sy, 1, -sy * w / 2],
    ])
    out = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

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
        transformed = (M @ np.hstack([corners, ones]).T).T
        rx1 = float(np.clip(transformed[:, 0].min() / w, 0.0, 1.0))
        ry1 = float(np.clip(transformed[:, 1].min() / h, 0.0, 1.0))
        rx2 = float(np.clip(transformed[:, 0].max() / w, 0.0, 1.0))
        ry2 = float(np.clip(transformed[:, 1].max() / h, 0.0, 1.0))
        nw, nh = rx2 - rx1, ry2 - ry1
        if nw > 0 and nh > 0:
            new_boxes.append(BBox(b.cls, (rx1 + rx2) / 2, (ry1 + ry2) / 2, nw, nh))
    return out, new_boxes
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -k "shear" -v 2>&1 | tail -10
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add apply_shear augmentation with bbox support"
```

---

### Task 5: Color / Texture Augmentations (Grayscale, Saturation, Exposure, Camera Gain)

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests**

Update import line, then append:

```python
# update import line:
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop, apply_shear,
    apply_grayscale, apply_saturation, apply_exposure, apply_camera_gain,
)


def test_grayscale_preserves_shape():
    img = np.array([[[100, 150, 200]]], dtype=np.uint8)
    result = apply_grayscale(img)
    assert result.shape == img.shape


def test_grayscale_all_channels_equal():
    img = np.array([[[50, 100, 200]]], dtype=np.uint8)
    result = apply_grayscale(img)
    assert int(result[0, 0, 0]) == int(result[0, 0, 1]) == int(result[0, 0, 2])


def test_saturation_preserves_shape():
    img = _solid_img(128)
    result = apply_saturation(img, 1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_saturation_zero_desaturates():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 200]
    result = apply_saturation(img, 0.0)
    # All channels should be equal after full desaturation
    assert int(result[0, 0, 0]) == int(result[0, 0, 1]) == int(result[0, 0, 2])


def test_exposure_preserves_shape():
    img = _solid_img(100)
    result = apply_exposure(img, 1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_exposure_clips_at_255():
    img = _solid_img(200)
    result = apply_exposure(img, 2.0)
    assert int(result.max()) <= 255


def test_camera_gain_increases_brightness():
    img = _solid_img(100)
    result = apply_camera_gain(img, 1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    # gain > 1 should not decrease brightness for a non-black image
    assert int(result.mean()) >= 100


def test_camera_gain_clips_at_255():
    img = _solid_img(200)
    result = apply_camera_gain(img, 2.0)
    assert int(result.max()) <= 255
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_grayscale_preserves_shape -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement the four functions in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
def apply_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_exposure(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_camera_gain(img: np.ndarray, gain: float) -> np.ndarray:
    result = img.astype(np.float32) * gain
    return np.clip(result, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -k "grayscale or saturation or exposure or camera_gain" -v 2>&1 | tail -12
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add grayscale, saturation, exposure, camera_gain augmentations"
```

---

### Task 6: Motion Blur Augmentation

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests for `apply_motion_blur`**

Update import line to add `apply_motion_blur`, then append:

```python
# update import line:
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop, apply_shear,
    apply_grayscale, apply_saturation, apply_exposure, apply_camera_gain,
    apply_motion_blur,
)


def test_motion_blur_preserves_shape():
    img = _solid_img()
    result = apply_motion_blur(img, 15, 0)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_motion_blur_even_kernel_accepted():
    img = _solid_img()
    result = apply_motion_blur(img, 4, 0)  # promoted to 5 internally
    assert result.shape == img.shape


def test_motion_blur_blurs_image():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, 25:] = 255  # half white
    result = apply_motion_blur(img, 15, 0)
    # The sharp edge should be smoothed - pixel at boundary won't be pure 0 or 255
    assert int(result[25, 20, 0]) != int(img[25, 20, 0]) or int(result[25, 30, 0]) != int(img[25, 30, 0])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_motion_blur_preserves_shape -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement `apply_motion_blur` in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
def apply_motion_blur(img: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    cx, cy = kernel_size / 2.0, kernel_size / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
    total = k.sum()
    if total > 0:
        k /= total
    return cv2.filter2D(img, -1, k)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -k "motion_blur" -v 2>&1 | tail -10
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add apply_motion_blur augmentation"
```

---

### Task 7: Mosaic Augmentation

**Files:**
- Modify: `augmentor/core/augment.py`
- Modify: `augmentor/tests/test_augment.py`

- [ ] **Step 1: Add failing tests for `apply_mosaic`**

Update import line to add `apply_mosaic`, then append:

```python
# update import line (final form):
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop, apply_shear,
    apply_grayscale, apply_saturation, apply_exposure, apply_camera_gain,
    apply_motion_blur, apply_mosaic,
)


def test_mosaic_preserves_shape():
    imgs = [_solid_img(i * 50) for i in range(4)]
    boxes_list = [[BBox(0, 0.5, 0.5, 0.3, 0.3)] for _ in range(4)]
    out_img, out_boxes = apply_mosaic(imgs, boxes_list)
    assert out_img.shape == imgs[0].shape
    assert out_img.dtype == np.uint8


def test_mosaic_returns_boxes():
    imgs = [_solid_img(i * 50) for i in range(4)]
    boxes_list = [[BBox(0, 0.5, 0.5, 0.3, 0.3)] for _ in range(4)]
    _, out_boxes = apply_mosaic(imgs, boxes_list)
    assert isinstance(out_boxes, list)


def test_mosaic_combines_distinct_colors():
    imgs = [_solid_img(0), _solid_img(255), _solid_img(128), _solid_img(64)]
    boxes_list = [[] for _ in range(4)]
    out_img, _ = apply_mosaic(imgs, boxes_list)
    # Canvas combines different values from 4 quadrants
    assert int(out_img.max()) != int(out_img.min())


def test_mosaic_boxes_normalized():
    imgs = [_solid_img(i * 50) for i in range(4)]
    boxes_list = [[BBox(0, 0.5, 0.5, 0.3, 0.3)] for _ in range(4)]
    _, out_boxes = apply_mosaic(imgs, boxes_list)
    for b in out_boxes:
        assert 0.0 <= b.x <= 1.0
        assert 0.0 <= b.y <= 1.0
        assert 0.0 < b.w <= 1.0
        assert 0.0 < b.h <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py::test_mosaic_preserves_shape -v 2>&1 | tail -5
```
Expected: `ImportError` or `FAILED`

- [ ] **Step 3: Implement `apply_mosaic` in `augment.py`**

Append to `augmentor/core/augment.py`:

```python
def apply_mosaic(
    imgs: List[np.ndarray],
    boxes_list: List[List[BBox]],
) -> Tuple[np.ndarray, List[BBox]]:
    """YOLOv5-style 4-image mosaic. imgs must contain exactly 4 images."""
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im for im in imgs]

    cx = random.randint(int(0.4 * w), int(0.6 * w))
    cy = random.randint(int(0.4 * h), int(0.6 * h))

    canvas = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    quad_regions = [
        (0, 0, cx, cy),
        (cx, 0, 2 * w, cy),
        (0, cy, cx, 2 * h),
        (cx, cy, 2 * w, 2 * h),
    ]

    canvas_boxes: List[BBox] = []
    for (x1, y1, x2, y2), img_i, boxes_i in zip(quad_regions, imgs, boxes_list):
        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            continue
        canvas[y1:y2, x1:x2] = cv2.resize(img_i, (pw, ph))
        for b in boxes_i:
            nx = (x1 + b.x * pw) / (2 * w)
            ny = (y1 + b.y * ph) / (2 * h)
            nw_c = b.w * pw / (2 * w)
            nh_c = b.h * ph / (2 * h)
            canvas_boxes.append(BBox(b.cls, nx, ny, nw_c, nh_c))

    crop_x1 = max(cx - w // 2, 0)
    crop_y1 = max(cy - h // 2, 0)
    crop_x2 = min(cx + w // 2, 2 * w)
    crop_y2 = min(cy + h // 2, 2 * h)
    out = canvas[crop_y1:crop_y2, crop_x1:crop_x2]
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h))

    crx1 = crop_x1 / (2 * w)
    cry1 = crop_y1 / (2 * h)
    crx2 = crop_x2 / (2 * w)
    cry2 = crop_y2 / (2 * h)
    cr_w = crx2 - crx1
    cr_h = cry2 - cry1

    new_boxes: List[BBox] = []
    for b in canvas_boxes:
        bx1 = b.x - b.w / 2
        by1 = b.y - b.h / 2
        bx2 = b.x + b.w / 2
        by2 = b.y + b.h / 2
        cx1_c = max(bx1, crx1)
        cy1_c = max(by1, cry1)
        cx2_c = min(bx2, crx2)
        cy2_c = min(by2, cry2)
        if cx2_c <= cx1_c or cy2_c <= cy1_c:
            continue
        orig_area = b.w * b.h
        if orig_area > 0 and (cx2_c - cx1_c) * (cy2_c - cy1_c) / orig_area < 0.3:
            continue
        nx1 = (cx1_c - crx1) / cr_w
        ny1 = (cy1_c - cry1) / cr_h
        nx2 = (cx2_c - crx1) / cr_w
        ny2 = (cy2_c - cry1) / cr_h
        new_boxes.append(BBox(b.cls, (nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1))
    return out, new_boxes
```

- [ ] **Step 4: Run all augment tests to confirm they pass**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_augment.py -v 2>&1 | tail -15
```
Expected: all tests pass (no failures)

- [ ] **Step 5: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/augment.py augmentor/tests/test_augment.py && git commit -m "feat: add apply_mosaic augmentation (YOLOv5-style 4-image)"
```

---

### Task 8: Pipeline Update (AugConfig + execution logic)

**Files:**
- Modify: `augmentor/core/pipeline.py`
- Modify: `augmentor/tests/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline test for mosaic**

Append to `augmentor/tests/test_pipeline.py`:

```python
def test_run_batch_mosaic_creates_files(tmp_path):
    images_dir, labels_dir = _make_dataset(tmp_path, n=4)
    config = AugConfig(mosaic_enabled=True, mosaic_pairs=3, multiplier=0)
    run_batch(images_dir, labels_dir, config, progress_cb=_noop)
    mosaic_imgs = list(images_dir.glob("*_mosaic_*.jpg"))
    assert len(mosaic_imgs) == 3


def test_collect_originals_excludes_mosaic(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "real.jpg").touch()
    (images_dir / "real_aug1.jpg").touch()
    (images_dir / "a_b_cutmix_1.jpg").touch()
    (images_dir / "a_b_c_d_mosaic_1.jpg").touch()
    result = _collect_originals(images_dir)
    assert len(result) == 1
    assert result[0].name == "real.jpg"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_pipeline.py::test_run_batch_mosaic_creates_files -v 2>&1 | tail -5
```
Expected: `FAILED` (AugConfig has no `mosaic_enabled`)

- [ ] **Step 3: Replace `pipeline.py` with the full updated version**

Replace the entire contents of `augmentor/core/pipeline.py`:

```python
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Callable, Tuple

import cv2
import numpy as np

from core.label import BBox, read_labels, write_labels
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop, apply_shear,
    apply_grayscale, apply_saturation, apply_exposure, apply_camera_gain,
    apply_motion_blur, apply_mosaic,
)


@dataclass
class AugConfig:
    # existing
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
    # new
    flip_h_enabled: bool = False
    flip_v_enabled: bool = False
    rotation_enabled: bool = False
    rotation_min: int = -45
    rotation_max: int = 45
    crop_enabled: bool = False
    crop_min: int = 70   # percent
    crop_max: int = 95   # percent
    shear_enabled: bool = False
    shear_min: int = -15
    shear_max: int = 15
    grayscale_enabled: bool = False
    saturation_enabled: bool = False
    saturation_min: int = 50   # percent of factor (50 = 0.5×)
    saturation_max: int = 150  # percent of factor (150 = 1.5×)
    exposure_enabled: bool = False
    exposure_min: int = 50
    exposure_max: int = 150
    motion_blur_enabled: bool = False
    motion_blur_kernel: int = 15
    motion_blur_angle_min: int = 0
    motion_blur_angle_max: int = 360
    camera_gain_enabled: bool = False
    camera_gain_min: int = 80   # percent (80 = 0.8×)
    camera_gain_max: int = 120  # percent (120 = 1.2×)
    mosaic_enabled: bool = False
    mosaic_pairs: int = 10


def _collect_originals(images_dir: Path) -> List[Path]:
    all_imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    result = []
    for p in all_imgs:
        if "_aug" not in p.stem and "_cutmix" not in p.stem and "_mosaic" not in p.stem:
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
    if config.flip_h_enabled:
        enabled.append("flip_h")
    if config.flip_v_enabled:
        enabled.append("flip_v")
    if config.rotation_enabled:
        enabled.append("rotation")
    if config.crop_enabled:
        enabled.append("crop")
    if config.shear_enabled:
        enabled.append("shear")
    if config.grayscale_enabled:
        enabled.append("grayscale")
    if config.saturation_enabled:
        enabled.append("saturation")
    if config.exposure_enabled:
        enabled.append("exposure")
    if config.motion_blur_enabled:
        enabled.append("motion_blur")
    if config.camera_gain_enabled:
        enabled.append("camera_gain")

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
        elif aug == "flip_h":
            img, boxes = apply_flip(img, boxes, horizontal=True, vertical=False)
        elif aug == "flip_v":
            img, boxes = apply_flip(img, boxes, horizontal=False, vertical=True)
        elif aug == "rotation":
            angle = random.uniform(config.rotation_min, config.rotation_max)
            img, boxes = apply_rotation(img, boxes, angle)
        elif aug == "crop":
            ratio = random.randint(config.crop_min, config.crop_max) / 100.0
            img, boxes = apply_crop(img, boxes, ratio)
        elif aug == "shear":
            sx = random.uniform(config.shear_min, config.shear_max)
            sy = random.uniform(config.shear_min, config.shear_max)
            img, boxes = apply_shear(img, boxes, sx, sy)
        elif aug == "grayscale":
            img = apply_grayscale(img)
        elif aug == "saturation":
            factor = random.randint(config.saturation_min, config.saturation_max) / 100.0
            img = apply_saturation(img, factor)
        elif aug == "exposure":
            factor = random.randint(config.exposure_min, config.exposure_max) / 100.0
            img = apply_exposure(img, factor)
        elif aug == "motion_blur":
            angle = random.randint(config.motion_blur_angle_min, config.motion_blur_angle_max)
            img = apply_motion_blur(img, config.motion_blur_kernel, angle)
        elif aug == "camera_gain":
            gain = random.randint(config.camera_gain_min, config.camera_gain_max) / 100.0
            img = apply_camera_gain(img, gain)

    return img, boxes


def preview_augment(
    img: np.ndarray,
    boxes: List[BBox],
    config: AugConfig,
) -> Tuple[np.ndarray, List[BBox]]:
    """Apply all enabled augmentations at midpoint values for live preview. Mosaic/CutMix skipped."""
    if config.brightness_enabled:
        delta = (config.brightness_min + config.brightness_max) // 2
        img = apply_brightness(img, delta)
    if config.blur_enabled:
        img = apply_blur(img, config.blur_radius)
    if config.noise_enabled:
        img = apply_noise(img, config.noise_strength)
    if config.scale_enabled:
        scale = (config.scale_min + config.scale_max) / 2.0
        img, boxes = apply_scale(img, boxes, scale)
    if config.flip_h_enabled:
        img, boxes = apply_flip(img, boxes, horizontal=True, vertical=False)
    if config.flip_v_enabled:
        img, boxes = apply_flip(img, boxes, horizontal=False, vertical=True)
    if config.rotation_enabled:
        angle = (config.rotation_min + config.rotation_max) / 2.0
        img, boxes = apply_rotation(img, boxes, angle)
    if config.crop_enabled:
        ratio = (config.crop_min + config.crop_max) / 2.0 / 100.0
        img, boxes = apply_crop(img, boxes, ratio)
    if config.shear_enabled:
        mid = (config.shear_min + config.shear_max) / 2.0
        img, boxes = apply_shear(img, boxes, mid, mid)
    if config.grayscale_enabled:
        img = apply_grayscale(img)
    if config.saturation_enabled:
        factor = (config.saturation_min + config.saturation_max) / 2.0 / 100.0
        img = apply_saturation(img, factor)
    if config.exposure_enabled:
        factor = (config.exposure_min + config.exposure_max) / 2.0 / 100.0
        img = apply_exposure(img, factor)
    if config.motion_blur_enabled:
        angle = (config.motion_blur_angle_min + config.motion_blur_angle_max) / 2.0
        img = apply_motion_blur(img, config.motion_blur_kernel, angle)
    if config.camera_gain_enabled:
        gain = (config.camera_gain_min + config.camera_gain_max) / 2.0 / 100.0
        img = apply_camera_gain(img, gain)
    return img, boxes


def run_batch(
    images_dir: Path,
    labels_dir: Path,
    config: AugConfig,
    progress_cb: Callable[[int, int], None],
) -> None:
    originals = _collect_originals(images_dir)
    cutmix_count = config.cutmix_pairs if config.cutmix_enabled else 0
    mosaic_count = config.mosaic_pairs if config.mosaic_enabled and len(originals) >= 4 else 0
    total = len(originals) * config.multiplier + cutmix_count + mosaic_count
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

    if config.mosaic_enabled and len(originals) >= 4:
        for i in range(config.mosaic_pairs):
            sampled = random.sample(originals, 4)
            imgs = [cv2.imread(str(p)) for p in sampled]
            boxes_list = [read_labels(labels_dir / (p.stem + ".txt")) for p in sampled]
            if any(im is None for im in imgs):
                continue
            mixed_img, mixed_boxes = apply_mosaic(imgs, boxes_list)
            out_stem = f"{sampled[0].stem}_mosaic_{i + 1}"
            cv2.imwrite(str(images_dir / (out_stem + sampled[0].suffix)), mixed_img)
            write_labels(labels_dir / (out_stem + ".txt"), mixed_boxes)
            done += 1
            progress_cb(done, total)
```

- [ ] **Step 4: Run all pipeline tests**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/test_pipeline.py -v 2>&1 | tail -15
```
Expected: all tests pass

- [ ] **Step 5: Run full test suite**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python -m pytest tests/ -v 2>&1 | tail -20
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/core/pipeline.py augmentor/tests/test_pipeline.py && git commit -m "feat: extend AugConfig and pipeline for 10 new augmentations"
```

---

### Task 9: GUI Update (Scrollable Settings + New Controls)

**Files:**
- Modify: `augmentor/gui/augment_tab.py`

Note: No automated tests for GUI. Verify manually by running `python main.py` and checking all toggles/sliders appear and live preview works.

- [ ] **Step 1: Replace `_build_settings` with scrollable version and add all new sections**

Replace the entire `_build_settings` method and update `_build_config` in `augmentor/gui/augment_tab.py`.

Replace the `_build_settings` method (lines 69-127):

```python
    def _build_settings(self, parent) -> None:
        outer = ttk.LabelFrame(parent, text="Settings", padding=4)
        outer.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0, width=260)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        frame = ttk.Frame(canvas, padding=4)
        frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(frame_id, width=event.width)

        frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(frame, text="images/ folder:").grid(row=0, column=0, sticky="w")
        self._images_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._images_var, width=22).grid(row=1, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_images).grid(row=1, column=1, padx=(4, 0))

        ttk.Label(frame, text="labels/ folder:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._labels_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._labels_var, width=22).grid(row=3, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_labels).grid(row=3, column=1, padx=(4, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

        row = 5
        row = self._aug_section(frame, row, "Brightness", "brightness",
                                [("min", -100, 100, -50), ("max", -100, 100, 50)])
        row = self._aug_section(frame, row, "Blur", "blur",
                                [("radius", 1, 15, 3)])
        row = self._aug_section(frame, row, "Noise", "noise",
                                [("strength", 0, 100, 25)])
        row = self._aug_section(frame, row, "Scale", "scale",
                                [("min ×", 50, 200, 80), ("max ×", 50, 200, 120)])
        row = self._aug_section(frame, row, "Flip H", "flip_h", [])
        row = self._aug_section(frame, row, "Flip V", "flip_v", [])
        row = self._aug_section(frame, row, "Rotation", "rotation",
                                [("min_deg", -180, 180, -45), ("max_deg", -180, 180, 45)])
        row = self._aug_section(frame, row, "Crop", "crop",
                                [("min_pct", 50, 99, 70), ("max_pct", 50, 99, 95)])
        row = self._aug_section(frame, row, "Shear", "shear",
                                [("min_deg", -45, 45, -15), ("max_deg", -45, 45, 15)])
        row = self._aug_section(frame, row, "Grayscale", "grayscale", [])
        row = self._aug_section(frame, row, "Saturation", "saturation",
                                [("min_pct", 0, 200, 50), ("max_pct", 0, 200, 150)])
        row = self._aug_section(frame, row, "Exposure", "exposure",
                                [("min_pct", 0, 200, 50), ("max_pct", 0, 200, 150)])
        row = self._aug_section(frame, row, "Motion Blur", "motion_blur",
                                [("kernel", 3, 31, 15), ("ang_min", 0, 360, 0), ("ang_max", 0, 360, 360)])
        row = self._aug_section(frame, row, "Camera Gain", "camera_gain",
                                [("min_pct", 0, 200, 80), ("max_pct", 0, 200, 120)])
        row = self._aug_section(frame, row, "CutMix", "cutmix",
                                [("pairs", 1, 100, 10)])
        row = self._aug_section(frame, row, "Mosaic", "mosaic",
                                [("pairs", 1, 100, 10)])

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        ttk.Label(frame, text="Multiplier (×):").grid(row=row, column=0, sticky="w")
        self._multiplier_var = tk.IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=50, textvariable=self._multiplier_var, width=6).grid(
            row=row, column=1, sticky="w")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(2, weight=0)
```

- [ ] **Step 2: Replace `_build_config` method**

Replace the `_build_config` method (lines 303-318):

```python
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
            flip_h_enabled=self._flip_h_enabled.get(),
            flip_v_enabled=self._flip_v_enabled.get(),
            rotation_enabled=self._rotation_enabled.get(),
            rotation_min=self._rotation_min_deg_var.get(),
            rotation_max=self._rotation_max_deg_var.get(),
            crop_enabled=self._crop_enabled.get(),
            crop_min=self._crop_min_pct_var.get(),
            crop_max=self._crop_max_pct_var.get(),
            shear_enabled=self._shear_enabled.get(),
            shear_min=self._shear_min_deg_var.get(),
            shear_max=self._shear_max_deg_var.get(),
            grayscale_enabled=self._grayscale_enabled.get(),
            saturation_enabled=self._saturation_enabled.get(),
            saturation_min=self._saturation_min_pct_var.get(),
            saturation_max=self._saturation_max_pct_var.get(),
            exposure_enabled=self._exposure_enabled.get(),
            exposure_min=self._exposure_min_pct_var.get(),
            exposure_max=self._exposure_max_pct_var.get(),
            motion_blur_enabled=self._motion_blur_enabled.get(),
            motion_blur_kernel=self._motion_blur_kernel_var.get(),
            motion_blur_angle_min=self._motion_blur_ang_min_var.get(),
            motion_blur_angle_max=self._motion_blur_ang_max_var.get(),
            camera_gain_enabled=self._camera_gain_enabled.get(),
            camera_gain_min=self._camera_gain_min_pct_var.get(),
            camera_gain_max=self._camera_gain_max_pct_var.get(),
            mosaic_enabled=self._mosaic_enabled.get(),
            mosaic_pairs=self._mosaic_pairs_var.get(),
        )
```

- [ ] **Step 3: Update `_load_result_pairs` to also match `_mosaic` files**

Find the `_load_result_pairs` method and replace the `aug_imgs` glob line:

```python
    def _load_result_pairs(self) -> None:
        if self._images_dir is None:
            return
        aug_imgs = sorted(
            p for p in self._images_dir.glob("*.jpg")
            if "_aug" in p.stem or "_cutmix" in p.stem or "_mosaic" in p.stem
        )
        self._result_pairs = []
        for aug_path in aug_imgs:
            aug_lbl = self._labels_dir / (aug_path.stem + ".txt")
            if "_cutmix" in aug_path.stem or "_mosaic" in aug_path.stem:
                orig_path = aug_path
                orig_lbl = aug_lbl
            else:
                orig_stem = aug_path.stem.rsplit("_aug", 1)[0]
                orig_path = self._images_dir / (orig_stem + aug_path.suffix)
                orig_lbl = self._labels_dir / (orig_stem + ".txt")
            self._result_pairs.append((orig_path, orig_lbl, aug_path, aug_lbl))

        self._results_mode = True
        self._current_idx = 0
        if self._result_pairs:
            self._show_result(0)
```

- [ ] **Step 4: Verify variable naming for slider-less sections**

For `flip_h`, `flip_v`, `grayscale` (no sliders), the `_aug_section` method creates no `_var` attributes (the loop over `params` is empty). Only `_flip_h_enabled`, `_flip_v_enabled`, `_grayscale_enabled` BooleanVars are created. Verify `_build_config` only references `self._flip_h_enabled.get()` etc. (no slider vars) — confirmed in Step 2 above.

- [ ] **Step 5: Run the GUI to verify manually**

```bash
cd /home/bws/colcon_ws/src/dataset_helper/augmentor && python main.py &
```

Check:
- Settings panel scrolls with mouse wheel
- All 17 toggle buttons visible (Brightness, Blur, Noise, Scale, Flip H, Flip V, Rotation, Crop, Shear, Grayscale, Saturation, Exposure, Motion Blur, Camera Gain, CutMix, Mosaic)
- Toggling any augmentation ON shows effect in live preview (except CutMix/Mosaic)
- Sliders update preview in real time

- [ ] **Step 6: Commit**

```bash
cd /home/bws/colcon_ws/src/dataset_helper && git add augmentor/gui/augment_tab.py && git commit -m "feat: scrollable settings panel with 10 new augmentation controls"
```

---

## Self-Review

**Spec coverage:**
- Flip H/V ✓ (Task 1)
- Rotation ✓ (Task 2)
- Crop ✓ (Task 3)
- Shear ✓ (Task 4)
- Grayscale, Saturation, Exposure, Camera Gain ✓ (Task 5)
- Motion Blur ✓ (Task 6)
- Mosaic (4-image YOLOv5-style) ✓ (Task 7)
- Pipeline integration ✓ (Task 8)
- GUI scrollable + new controls ✓ (Task 9)

**Variable naming consistency:**
- `_motion_blur_kernel_var` (kernel slider, no `x` suffix) — used consistently in Task 8 AugConfig fields and Task 9 `_build_config`
- `_motion_blur_ang_min_var` / `_motion_blur_ang_max_var` — `_aug_section` generates these from param names `"ang min"` and `"ang max"` → `ang_min` and `ang_max` after `.replace(' ', '_')` → attr names `_motion_blur_ang_min_var` / `_motion_blur_ang_max_var`
- `_rotation_min_x_var` etc — param name is `"min °"`, `.replace(' ', '_').replace('×','x')` → `min_°_var`... **BUG FOUND**: `°` is not handled by `.replace('×','x')` so the attr name becomes `_rotation_min_°_var` which is not a valid Python identifier.

**Fix applied in plan:** Task 9 uses safe ASCII param names throughout:
- Rotation/Shear: `"min_deg"` / `"max_deg"` → `_xxx_min_deg_var` / `_xxx_max_deg_var`
- Crop/Saturation/Exposure/Camera Gain: `"min_pct"` / `"max_pct"` → `_xxx_min_pct_var` / `_xxx_max_pct_var`
- Motion Blur: `"ang_min"` / `"ang_max"` (no spaces) → `_motion_blur_ang_min_var` / `_motion_blur_ang_max_var`

All `_build_config` references corrected to match. No action needed — fixed above.
