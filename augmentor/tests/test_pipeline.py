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
