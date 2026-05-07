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
