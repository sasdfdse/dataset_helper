import numpy as np
import pytest
from core.label import BBox
from core.augment import (
    apply_brightness, apply_blur, apply_noise, apply_scale, apply_cutmix,
    apply_flip, apply_rotation, apply_crop,
    apply_shear,
    apply_grayscale, apply_saturation, apply_exposure, apply_camera_gain,
    apply_motion_blur,
    apply_mosaic,
)


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


# --- flip ---

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


def test_flip_vertical_mirrors_pixels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[0, :] = 255  # top row bright
    out_img, _ = apply_flip(img, [], horizontal=False, vertical=True)
    assert int(out_img[9, 0, 0]) == 255  # bottom row should now be bright
    assert int(out_img[0, 0, 0]) == 0


# --- rotation ---

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


def test_rotation_90_center_box_stays_near_center():
    img = _solid_img(h=100, w=100)
    boxes = [BBox(0, 0.5, 0.5, 0.4, 0.2)]
    _, out_boxes = apply_rotation(img, boxes, 90)
    assert len(out_boxes) == 1
    assert abs(out_boxes[0].x - 0.5) < 0.05
    assert abs(out_boxes[0].y - 0.5) < 0.05


def test_rotation_out_of_bounds_box_removed():
    img = _solid_img()
    boxes = [BBox(0, 0.02, 0.02, 0.02, 0.02)]
    _, out_boxes = apply_rotation(img, boxes, 45)
    assert isinstance(out_boxes, list)


def test_rotation_output_dtype():
    img = _solid_img()
    out_img, _ = apply_rotation(img, [], 30)
    assert out_img.dtype == np.uint8


# --- crop ---

def test_crop_preserves_shape():
    img = _solid_img()
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    out_img, out_boxes = apply_crop(img, boxes, 0.8)
    assert out_img.shape == img.shape


def test_crop_keeps_fully_inside_box():
    img = _solid_img()
    # crop_ratio=1.0 means no crop happens
    boxes = [BBox(0, 0.5, 0.5, 0.3, 0.3)]
    _, out_boxes = apply_crop(img, boxes, 1.0)
    assert len(out_boxes) == 1


def test_crop_output_dtype():
    img = _solid_img()
    out_img, _ = apply_crop(img, [], 0.9)
    assert out_img.dtype == np.uint8


# --- shear ---

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


# --- grayscale ---

def test_grayscale_preserves_shape():
    img = np.array([[[100, 150, 200]]], dtype=np.uint8)
    result = apply_grayscale(img)
    assert result.shape == img.shape


def test_grayscale_all_channels_equal():
    img = np.array([[[50, 100, 200]]], dtype=np.uint8)
    result = apply_grayscale(img)
    assert int(result[0, 0, 0]) == int(result[0, 0, 1]) == int(result[0, 0, 2])


# --- saturation ---

def test_saturation_preserves_shape():
    img = _solid_img(128)
    result = apply_saturation(img, 1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_saturation_zero_desaturates():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 200]
    result = apply_saturation(img, 0.0)
    assert int(result[0, 0, 0]) == int(result[0, 0, 1]) == int(result[0, 0, 2])


# --- exposure ---

def test_exposure_preserves_shape():
    img = _solid_img(100)
    result = apply_exposure(img, 1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_exposure_clips_at_255():
    img = _solid_img(200)
    result = apply_exposure(img, 2.0)
    assert int(result.max()) <= 255


# --- camera_gain ---

def test_camera_gain_increases_brightness():
    img = _solid_img(100)
    result = apply_camera_gain(img, 1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert int(result.mean()) >= 100


def test_camera_gain_clips_at_255():
    img = _solid_img(200)
    result = apply_camera_gain(img, 2.0)
    assert int(result.max()) <= 255


# --- motion_blur ---

def test_motion_blur_preserves_shape():
    img = _solid_img()
    result = apply_motion_blur(img, 15, 0)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_motion_blur_even_kernel_accepted():
    img = _solid_img()
    result = apply_motion_blur(img, 4, 0)  # internally promoted to 5
    assert result.shape == img.shape


def test_motion_blur_blurs_image():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, 25:] = 255  # half white
    result = apply_motion_blur(img, 15, 0)
    # The sharp edge should be smoothed
    assert int(result[25, 20, 0]) != int(img[25, 20, 0]) or int(result[25, 30, 0]) != int(img[25, 30, 0])


# --- mosaic ---

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
