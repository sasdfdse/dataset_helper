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
    if config.cutmix_enabled:
        cutmix_count = config.cutmix_pairs
    else:
        cutmix_count = 0
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
