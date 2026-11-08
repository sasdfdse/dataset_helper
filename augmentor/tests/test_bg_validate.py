import numpy as np
import pytest

from background_swap.common.domain import Detection
from background_swap.stages.validate import (
    VLM_PROMPT,
    combine,
    find_new_objects,
    iou,
    parse_vlm_json,
    rule_checks,
)


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_rule_checks_ok():
    orig = np.zeros((10, 10, 3), np.uint8); out = orig.copy(); out[:, 5:] = 100
    protect = np.zeros((10, 10), bool); protect[:, :5] = True
    reasons, checks = rule_checks(orig, out, protect, ~protect, 20.0)
    assert reasons == [] and checks["protected_identical"] and checks["mean_change"] == 100.0


def test_rule_checks_detects_protected_change_and_little_change():
    orig = np.zeros((10, 10, 3), np.uint8); out = orig.copy(); out[0, 0] = 1; out[:, 5:] = 5
    protect = np.zeros((10, 10), bool); protect[:, :5] = True
    assert rule_checks(orig, out, protect, ~protect, 20.0)[0] == ["protected_changed", "too_little_change"]


def test_rule_checks_size_mismatch_short_circuits():
    reasons, checks = rule_checks(np.zeros((10, 10, 3), np.uint8), np.zeros((9, 10, 3), np.uint8),
                                  np.zeros((10, 10), bool), np.ones((10, 10), bool), 20.0)
    assert reasons == ["size_mismatch"]
    assert checks == {"protected_identical": False, "mean_change": 0.0}


def test_rule_checks_no_uint8_wraparound():
    orig = np.full((4, 4, 3), 250, np.uint8); out = np.zeros((4, 4, 3), np.uint8)
    inpaint = np.ones((4, 4), bool)
    reasons, checks = rule_checks(orig, out, np.zeros((4, 4), bool), inpaint, 20.0)
    assert checks["mean_change"] == 250.0 and reasons == []


def test_rule_checks_single_channel_protected_diff_and_empty_inpaint():
    orig = np.zeros((4, 4, 3), np.uint8); out = orig.copy(); out[1, 1, 2] = 1
    protect = np.zeros((4, 4), bool); protect[1, 1] = True
    reasons, checks = rule_checks(orig, out, protect, np.zeros((4, 4), bool), 20.0)
    assert checks["protected_identical"] is False
    assert checks["mean_change"] == 0.0
    assert reasons == ["protected_changed", "too_little_change"]


def test_find_new_objects_filters_by_iou_score_and_region():
    inpaint = np.zeros((100, 100), bool); inpaint[:, 50:] = True
    labels = [(0, 0, 20, 20)]
    dets = [Detection((0, 0, 20, 20), 0.9, "robot"),     # overlaps existing label
            Detection((60, 60, 80, 80), 0.3, "robot"),   # low score
            Detection((10, 60, 30, 80), 0.9, "robot"),   # protected region
            Detection((60, 10, 80, 30), 0.9, "robot")]   # new object
    assert find_new_objects(dets, labels, inpaint) == [dets[3]]


def test_find_new_objects_zero_area_clipping_and_order():
    inpaint = np.ones((50, 50), bool)
    dets = [Detection((10, 10, 10, 30), 0.9, "robot"),    # zero area
            Detection((40, 40, 90, 90), 0.9, "robot"),    # clipped, still inside
            Detection((60, 60, 90, 90), 0.9, "robot"),    # fully outside the image
            Detection((1, 1, 5, 5), 0.9, "robot")]
    assert find_new_objects(dets, [], inpaint) == [dets[1], dets[3]]


@pytest.mark.parametrize("text,ok", [
    ('Sure: {"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": false, "reason": "ok"}', True),
    ('{"natural_indoor_venue": "yes", "boundary_artifacts": false, "new_objects": false}', False),
    ('{"natural_indoor_venue": 1, "boundary_artifacts": false, "new_objects": false}', False),
    ('{"natural_indoor_venue": true, "boundary_artifacts": false}', False),
    ('no json here', False)])
def test_parse_vlm_json(text, ok):
    assert (parse_vlm_json(text) is not None) is ok


def test_parse_vlm_json_nested_braces_and_retry():
    text = ('example {not json} then '
            '{"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": true, '
            '"reason": "a } brace {"}')
    d = parse_vlm_json(text)
    assert d is not None and d["new_objects"] is True and d["reason"] == "a } brace {"
    assert parse_vlm_json('{"a": {"b": 1}') is None  # unbalanced


def test_vlm_prompt_format():
    out = VLM_PROMPT.format(class_names="robot")
    assert "robot" in out and "natural_indoor_venue" in out and "{class_names}" not in out


def test_combine_orders_reasons_and_ors_new_objects():
    vlm = '{"natural_indoor_venue": false, "boundary_artifacts": true, "new_objects": true, "reason": "x"}'
    r = combine([], {}, [Detection((0, 0, 1, 1), 0.9, "robot")], vlm)
    assert not r.accepted
    assert r.reasons == ["new_object:gdino", "vlm_unnatural", "vlm_boundary_artifacts", "new_object:vlm"]


def test_combine_accepts_when_all_skipped_or_clean():
    assert combine([], {}, None, None).accepted
    assert combine([], {}, [], '{"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": false}').accepted


def test_combine_unparsable_vlm_rejects():
    r = combine([], {}, None, "garbage")
    assert r.reasons == ["vlm_unparsable"]
    assert r.checks["vlm"] == "garbage" and r.checks["gdino_new_count"] is None


def test_combine_checks_and_rule_reasons_first():
    base = {"mean_change": 1.0}
    r = combine(["too_little_change"], base, [Detection((0, 0, 1, 1), 0.9, "robot")], None)
    assert r.reasons == ["too_little_change", "new_object:gdino"]
    assert r.checks == {"mean_change": 1.0, "gdino_new_count": 1, "vlm": None}
    assert base == {"mean_change": 1.0}  # input not mutated


def test_vlm_prompt_does_not_prime_synthetic():
    low = VLM_PROMPT.format(class_names="robot").lower()
    for word in ("synthetic", "generated", "replaced", "composited"):
        assert word not in low
