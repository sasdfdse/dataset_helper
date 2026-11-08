from pathlib import Path

import cv2
import numpy as np
import pytest

from background_swap.dataset.classes import find_class_names_file, load_class_names
from background_swap.dataset.inputs import collect_items, load_image_bgr, looks_non_train


def _dataset(tmp_path: Path, n: int):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(n):
        cv2.imwrite(str(images_dir / f"img_{i:03d}.jpg"), np.full((10, 10, 3), 100, np.uint8))
        (labels_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return images_dir, labels_dir


def test_collect_items_frame_stride(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=6)
    items, skipped = collect_items(imgs, lbls, frame_stride=2)
    assert [i.stem for i in items] == ["img_000", "img_002", "img_004"] and skipped == []
    assert (items[0].width, items[0].height) == (10, 10)


def test_collect_items_skips_missing_label_and_bad_image(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=2)
    (lbls / "img_001.txt").unlink()
    (imgs / "broken.jpg").write_bytes(b"x")
    (lbls / "broken.txt").write_text("")
    items, skipped = collect_items(imgs, lbls, 1)
    assert [i.stem for i in items] == ["img_000"]
    assert sorted(skipped) == [("broken", "unreadable_image"), ("img_001", "missing_label")]


def test_collect_items_duplicate_stem(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=1)
    cv2.imwrite(str(imgs / "img_000.png"), np.zeros((10, 10, 3), np.uint8))
    items, skipped = collect_items(imgs, lbls, 1)
    assert len(items) == 1 and skipped == [("img_000", "duplicate_stem")]


def test_collect_items_excludes_bgswap(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=1)
    cv2.imwrite(str(imgs / "img_000_bgswap1.jpg"), np.zeros((10, 10, 3), np.uint8))
    assert [i.stem for i in collect_items(imgs, lbls, 1)[0]] == ["img_000"]


def test_load_image_bgr_converts_rgba_and_gray(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 4), np.uint8))
    cv2.imwrite(str(tmp_path / "g.png"), np.zeros((8, 8), np.uint8))
    assert load_image_bgr(tmp_path / "a.png").shape == (8, 8, 3)
    assert load_image_bgr(tmp_path / "g.png").shape == (8, 8, 3)


@pytest.mark.parametrize("p,expected", [("data/val/images", True), ("data/test", True),
    ("data/train/images", False), ("data/valley/images", False), ("Validation/x", True)])
def test_looks_non_train(p, expected):
    assert looks_non_train(Path(p)) is expected


def test_find_class_names_prefers_classes_txt_in_labels(tmp_path):
    (tmp_path / "lbl").mkdir()
    (tmp_path / "img").mkdir()
    (tmp_path / "data.yaml").write_text("names: [a, b]")
    (tmp_path / "lbl" / "classes.txt").write_text("x\ny\n")
    assert find_class_names_file(tmp_path / "img", tmp_path / "lbl") == tmp_path / "lbl" / "classes.txt"


def test_find_class_names_folder_order_beats_name_order(tmp_path):
    (tmp_path / "lbl").mkdir()
    (tmp_path / "img").mkdir()
    (tmp_path / "lbl" / "obj.names").write_text("x\n")
    (tmp_path / "classes.txt").write_text("y\n")
    assert find_class_names_file(tmp_path / "img", tmp_path / "lbl") == tmp_path / "lbl" / "obj.names"


def test_find_class_names_none(tmp_path):
    (tmp_path / "lbl").mkdir()
    (tmp_path / "img").mkdir()
    assert find_class_names_file(tmp_path / "img", tmp_path / "lbl") is None


def test_load_class_names_yaml_dict(tmp_path):
    (tmp_path / "data.yaml").write_text("names:\n  1: ball\n  0: robot\n")
    assert load_class_names(tmp_path / "data.yaml") == ["robot", "ball"]


def test_load_class_names_yaml_list_and_missing(tmp_path):
    (tmp_path / "a.yaml").write_text("names: [a, b]")
    (tmp_path / "b.yaml").write_text("nc: 2")
    assert load_class_names(tmp_path / "a.yaml") == ["a", "b"]
    assert load_class_names(tmp_path / "b.yaml") == []


def test_load_class_names_txt(tmp_path):
    (tmp_path / "classes.txt").write_text(" a \n\nb\n  \n")
    assert load_class_names(tmp_path / "classes.txt") == ["a", "b"]


def test_load_class_names_yaml_non_int_keys_returns_empty(tmp_path):
    (tmp_path / "data.yaml").write_text("names: {a: x}\n")
    assert load_class_names(tmp_path / "data.yaml") == []


def test_load_class_names_malformed_yaml_returns_empty(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [a, b\n  : : {\n")
    assert load_class_names(tmp_path / "data.yaml") == []


def test_load_class_names_non_utf8_returns_empty(tmp_path):
    (tmp_path / "classes.txt").write_bytes(b"\xff\xfe\x80bad\n")
    (tmp_path / "data.yaml").write_bytes(b"names: [\xff\xfe]\n")
    assert load_class_names(tmp_path / "classes.txt") == []
    assert load_class_names(tmp_path / "data.yaml") == []
