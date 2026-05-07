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
