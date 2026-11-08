import numpy as np

from background_swap.stages.masks import (bbox_to_xyxy, build_mask_set, keep_item_mask,
                                   needs_sam, object_core_mask, object_protect_mask)
from background_swap.common.domain import Detection
from background_swap.common.settings import BackgroundSwapSettings, KeepItem
from core.label import BBox
from tests.bg_fakes import (GREEN_HSV_RANGE, FakeDetector, FakeSegmenter,
                            field_image)


def _img():
    return np.full((100, 100, 3), 128, np.uint8)


def _field_img():
    return field_image()


def test_small_box_uses_rect_without_sam():
    b = BBox(0, 0.5, 0.5, 0.1, 0.1)
    assert not needs_sam(b, 100, 100, 32)
    core = object_core_mask(100, 100, [b], [None], 32)
    assert core[50, 50] and core[44, 44] and not core[40, 40]


def test_sam_mask_too_small_falls_back_to_rect():
    b = BBox(0, 0.5, 0.5, 0.5, 0.5)
    tiny = np.zeros((100, 100), bool); tiny[50, 50] = True
    core = object_core_mask(100, 100, [b], [tiny], 32)
    assert core[30, 30]


def test_sam_mask_clipped_to_expanded_box():
    b = BBox(0, 0.5, 0.5, 0.4, 0.4)
    big = np.ones((100, 100), bool)
    core = object_core_mask(100, 100, [b], [big], 32)
    assert core[50, 50] and not core[5, 5]


def test_object_core_clips_and_ignores_degenerate_boxes():
    boxes = [BBox(0, 0.95, 0.5, 0.3, 0.2), BBox(0, 0.5, 0.5, 0.0, 0.2)]
    core = object_core_mask(100, 100, boxes, [None, None], 32)
    assert core.shape == (100, 100) and core[50, 99] and not core[50, 50]


def test_protect_dilation():
    b = BBox(0, 0.5, 0.4, 0.2, 0.2)
    core = object_core_mask(100, 100, [b], [None], 32)
    assert object_protect_mask(core, [b], dilate_px=6, shadow_extend_ratio=0.0)[40, 64]
    assert not object_protect_mask(core, [b], dilate_px=0, shadow_extend_ratio=0.0)[40, 64]


def test_protect_shadow_extension():
    b = BBox(0, 0.5, 0.4, 0.2, 0.2)
    core = object_core_mask(100, 100, [b], [None], 32)
    p = object_protect_mask(core, [b], dilate_px=0, shadow_extend_ratio=0.15)
    assert p[52, 39] and not p[52, 30] and not p[58, 50]


def test_build_mask_set_required_keep_missing():
    s = BackgroundSwapSettings(keep_items=[KeepItem("fence", "text", required=True, prompt="fence")])
    ms = build_mask_set(_img(), [], s, FakeSegmenter(), FakeDetector({}))
    assert ms.skip_reason == "required_keep_missing:fence"


def test_build_mask_set_optional_keep_missing_warns():
    s = BackgroundSwapSettings(keep_items=[KeepItem("fence", "text", prompt="fence")])
    ms = build_mask_set(_img(), [], s, FakeSegmenter(), FakeDetector({}))
    assert ms.skip_reason is None and ms.warnings == ["keep_not_found:fence"]


def test_build_mask_set_inpaint_is_complement_of_protect():
    lo, hi = GREEN_HSV_RANGE
    s = BackgroundSwapSettings(keep_items=[KeepItem("field", "color", hsv_lower=lo, hsv_upper=hi)])
    ms = build_mask_set(_field_img(), [BBox(0, 0.5, 0.5, 0.2, 0.2)], s, FakeSegmenter(), None)
    assert ms.keep[90, 50] and not ms.keep[5, 50]
    assert (ms.inpaint == ~(ms.object_protect | ms.keep)).all()


def test_build_mask_set_small_inpaint_area_skips():
    ms = build_mask_set(_img(), [BBox(0, 0.5, 0.5, 0.98, 0.98)], BackgroundSwapSettings(), FakeSegmenter(), None)
    assert ms.skip_reason == "small_inpaint_area"


def test_build_mask_set_no_boxes():
    ms = build_mask_set(_img(), [], BackgroundSwapSettings(), None, None)
    assert not ms.object_core.any() and ms.inpaint.all() and ms.skip_reason is None


def test_segmenter_called_once_for_large_boxes_only():
    seg = FakeSegmenter()
    build_mask_set(_img(), [BBox(0, .3, .3, .4, .4), BBox(0, .8, .8, .05, .05)], BackgroundSwapSettings(), seg, None)
    assert len(seg.calls) == 1 and len(seg.calls[0]) == 1


# --- 추가 테스트 (controller rulings) ---

def test_keep_text_unions_segmented_boxes_and_uses_thresholds():
    det = FakeDetector({"fence": [Detection((10, 10, 30, 30), 0.9, "fence"),
                                  Detection((60, 60, 80, 80), 0.8, "fence")]})
    seg = FakeSegmenter()
    m, ok = keep_item_mask(KeepItem("fence", "text", prompt="fence"), _img(), seg, det)
    assert ok and m[20, 20] and m[70, 70] and not m[45, 45]
    assert det.calls == [(["fence"], 0.35, 0.25)]


def test_keep_text_without_segmenter_or_detector_fails():
    item = KeepItem("fence", "text", prompt="fence")
    det = FakeDetector({"fence": [Detection((10, 10, 30, 30), 0.9, "fence")]})
    assert keep_item_mask(item, _img(), None, det)[1] is False
    assert keep_item_mask(item, _img(), FakeSegmenter(), None)[1] is False


def test_keep_color_below_min_area_fails():
    lo, hi = GREEN_HSV_RANGE
    item = KeepItem("field", "color", hsv_lower=lo, hsv_upper=hi, min_area_ratio=0.9)
    m, ok = keep_item_mask(item, _field_img(), None, None)
    assert not ok


def test_required_color_keep_missing_skips_with_complement_inpaint():
    lo, hi = GREEN_HSV_RANGE
    s = BackgroundSwapSettings(keep_items=[KeepItem("field", "color", required=True, hsv_lower=lo, hsv_upper=hi)])
    ms = build_mask_set(_img(), [], s, None, None)
    assert ms.skip_reason == "required_keep_missing:field"
    assert (ms.inpaint == ~(ms.object_protect | ms.keep)).all()


# --- fix round 1: bbox_to_xyxy / shadow extent / SAM rules ---

def test_bbox_to_xyxy_exact_margin_zero_and_tenth():
    b = BBox(0, .5, .5, .1, .1)
    assert bbox_to_xyxy(b, 100, 100) == (45, 45, 55, 55)
    assert bbox_to_xyxy(b, 100, 100, margin=0.1) == (44, 44, 56, 56)


def test_bbox_to_xyxy_non_integer_uses_floor_and_ceil():
    # x: 33.3 +- 5 -> 28.3..38.3 -> floor 28, ceil 39
    assert bbox_to_xyxy(BBox(0, .333, .5, .1, .1), 100, 100) == (28, 45, 39, 55)
    # non-square image: w=200 h=50, x 0.25*200=50 +-10 ; y 0.5*50=25 +-2.5 -> 22.5..27.5
    assert bbox_to_xyxy(BBox(0, .25, .5, .1, .1), 200, 50) == (40, 22, 60, 28)


def test_bbox_to_xyxy_clips_at_image_edges():
    assert bbox_to_xyxy(BBox(0, .95, .5, .3, .2), 100, 100) == (80, 40, 100, 60)
    assert bbox_to_xyxy(BBox(0, .02, .02, .2, .2), 100, 100) == (0, 0, 12, 12)
    assert bbox_to_xyxy(BBox(0, .5, .5, 2.0, 2.0), 100, 100) == (0, 0, 100, 100)


def test_bbox_to_xyxy_none_for_degenerate_or_off_image():
    assert bbox_to_xyxy(BBox(0, .5, .5, 0.0, .2), 100, 100) is None
    assert bbox_to_xyxy(BBox(0, .5, .5, .2, 0.0), 100, 100) is None
    assert bbox_to_xyxy(BBox(0, 1.5, .5, .1, .1), 100, 100) is None
    assert bbox_to_xyxy(BBox(0, .5, -0.5, .1, .1), 100, 100) is None


def test_bbox_to_xyxy_returns_python_ints():
    r = bbox_to_xyxy(BBox(0, .333, .5, .1, .1), 100, 100, margin=0.1)
    assert r is not None and all(type(v) is int for v in r)


def test_protect_shadow_exact_extent():
    b = BBox(0, 0.5, 0.4, 0.2, 0.2)          # x 40..60, y 30..50
    # empty core isolates the shadow strip itself
    strip = object_protect_mask(np.zeros((100, 100), bool), [b], dilate_px=0, shadow_extend_ratio=0.15)
    ys, xs = np.where(strip)
    # half-open x 38..62, y 50..53 -> inclusive 38..61 / 50..52
    assert (xs.min(), xs.max()) == (38, 61)
    assert (ys.min(), ys.max()) == (50, 52)
    assert strip.sum() == 24 * 3               # full rectangle, nothing else

    # with the real core (rect margin .1 -> y 28..52) only row 52 lies outside core
    core = object_core_mask(100, 100, [b], [None], 32)
    p = object_protect_mask(core, [b], dilate_px=0, shadow_extend_ratio=0.15)
    ys, xs = np.where(p & ~core)
    assert (xs.min(), xs.max()) == (38, 61) and set(ys) == {52}
    assert not p[53, 50] and not p[52, 37] and not p[52, 62]


def test_segmenter_receives_margin_zero_float_xyxy_of_large_box():
    seg = FakeSegmenter()
    build_mask_set(_img(), [BBox(0, .3, .3, .4, .4), BBox(0, .8, .8, .05, .05)],
                   BackgroundSwapSettings(), seg, None)
    assert seg.calls[0] == [(10.0, 10.0, 50.0, 50.0)]
    assert all(type(v) is float for v in seg.calls[0][0])


def test_needs_sam_box_without_segmenter_falls_back_to_rect():
    ms = build_mask_set(_img(), [BBox(0, .5, .5, .5, .5)], BackgroundSwapSettings(), None, None)
    assert ms.object_core[20, 20] and ms.object_core[79, 79]
    assert not ms.object_core[19, 19] and not ms.object_core[80, 80]


def test_sam_mask_keeps_largest_component_only():
    b = BBox(0, .5, .5, .5, .5)               # rect (margin .1): 20..80
    sam = np.zeros((100, 100), bool)
    sam[30:60, 30:60] = True                   # blob 900 px
    sam[75, 75] = True                         # distant speck inside the rect
    core = object_core_mask(100, 100, [b], [sam], 32)
    assert core[40, 40] and not core[75, 75] and not core[25, 25]
    assert core.sum() == 900


def test_sam_mask_area_threshold_boundary():
    b = BBox(0, .5, .5, .5, .5)               # bbox 50x50 = 2500 -> threshold 500
    at = np.zeros((100, 100), bool)
    at[30:55, 30:50] = True                    # exactly 500 -> kept
    core = object_core_mask(100, 100, [b], [at], 32)
    assert core.sum() == 500 and not core[25, 25]
    below = at.copy()
    below[30, 30] = False                      # 499 -> rect fallback
    core = object_core_mask(100, 100, [b], [below], 32)
    assert core[25, 25] and core.sum() == 60 * 60
