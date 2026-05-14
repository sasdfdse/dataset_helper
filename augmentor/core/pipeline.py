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
    crop_min: int = 70
    crop_max: int = 95
    shear_enabled: bool = False
    shear_min: int = -15
    shear_max: int = 15
    grayscale_enabled: bool = False
    saturation_enabled: bool = False
    saturation_min: int = 50
    saturation_max: int = 150
    exposure_enabled: bool = False
    exposure_min: int = 50
    exposure_max: int = 150
    motion_blur_enabled: bool = False
    motion_blur_kernel: int = 15
    motion_blur_angle_min: int = 0
    motion_blur_angle_max: int = 360
    camera_gain_enabled: bool = False
    camera_gain_min: int = 80
    camera_gain_max: int = 120
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
