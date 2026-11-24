import time

import pytest

from background_swap.common.estimate import (EtaTracker, describe_plan, format_duration,
                                             format_finish, inpaint_seconds, make_plan)
from background_swap.common.settings import BackgroundSwapSettings


def test_inpaint_seconds_matches_measurements():
    s = BackgroundSwapSettings()
    # a 640x480 photo is drawn at 1024x768 in auto mode (measured 19.7s)
    assert inpaint_seconds(640, 480, s) == pytest.approx(19.7, abs=0.5)
    manual = BackgroundSwapSettings(resolution_mode="manual", manual_resolution_px=768)
    assert inpaint_seconds(640, 480, manual) == pytest.approx(12.7, abs=1.5)    # 768x576 measured 12.7s
    assert inpaint_seconds(1280, 720, s) == pytest.approx(23.7, abs=1.5)


def test_plan_scales_with_images_variants_and_options():
    s = BackgroundSwapSettings(num_variants=3)
    plan = make_plan(s, 100, (640, 480))
    assert (plan.images, plan.jobs) == (100, 300)
    one = make_plan(BackgroundSwapSettings(num_variants=1), 100, (640, 480))
    assert one.total_s < plan.total_s / 2.5
    fewer = make_plan(BackgroundSwapSettings(num_variants=1), 10, (640, 480))
    assert fewer.total_s < one.total_s / 5
    no_vlm = make_plan(BackgroundSwapSettings(num_variants=1, use_vlm=False), 100, (640, 480))
    assert no_vlm.total_s < one.total_s and no_vlm.per_image_s < one.per_image_s
    assert make_plan(s, 0, None).total_s == 0


def test_format_duration():
    assert format_duration(30) == "1분 미만"
    assert format_duration(5 * 60) == "약 5분"
    assert format_duration(2 * 3600 + 10 * 60) == "약 2시간 10분"
    assert format_duration(3600) == "약 1시간"
    assert format_duration(106 * 3600) == "약 4일 10시간"


def test_format_finish_adds_date_only_when_not_today():
    now = time.mktime((2026, 11, 15, 10, 0, 0, 0, 0, -1))
    assert format_finish(90 * 60, now) == "11:30"
    assert format_finish(20 * 3600, now) == "11/16 06:00"


def test_describe_plan():
    s = BackgroundSwapSettings(num_variants=3, max_attempts=3)
    text = describe_plan(make_plan(s, 637, (640, 480)), s)
    assert "637장 × 변형 3 = 1,911장" in text and "재시도 시 최대 ×3" in text and "시간" in text
    assert "선택" in describe_plan(make_plan(s, 0, None), s)


def test_eta_tracker_uses_measured_speed_and_adds_later_stages():
    s = BackgroundSwapSettings(num_variants=1, use_llm=False, use_vlm=False)
    plan = make_plan(s, 10, (640, 480))
    eta = EtaTracker(plan)
    # before the first item: use the estimate
    first = eta.update("inpaint", 0, 10, now=0.0)
    assert first == pytest.approx(10 * plan.stages["inpaint"].per_item_s + plan.after("inpaint"), rel=0.01)
    # skip the first item (includes load) and use the measured speed after it (5s/item)
    eta.update("inpaint", 1, 10, now=60.0)
    rem = eta.update("inpaint", 3, 10, now=70.0)
    validate = plan.stages["validate"]
    assert rem == pytest.approx(7 * 5.0 + validate.load_s + validate.per_item_s * 10)
    # a new stage measures its own speed
    assert eta.update("validate", 0, 10, now=80.0) == pytest.approx(10 * validate.per_item_s)


def test_eta_tracker_without_plan_waits_for_speed():
    eta = EtaTracker()
    assert eta.update("inpaint", 0, 4, now=0.0) is None
    assert eta.update("inpaint", 1, 4, now=10.0) is None
    assert eta.update("inpaint", 2, 4, now=14.0) == pytest.approx(8.0)


def test_fast_mode_makes_inpaint_faster():
    normal = make_plan(BackgroundSwapSettings(num_variants=1), 100, (640, 480))
    fast = make_plan(BackgroundSwapSettings(num_variants=1, fast_mode=True), 100, (640, 480))
    assert fast.stages["inpaint"].per_item_s == pytest.approx(6.9, abs=0.5)   # measured at 1024x768
    assert fast.total_s < normal.total_s / 2


def test_reference_photos_add_a_stage(tmp_path):
    from pathlib import Path
    s = BackgroundSwapSettings(num_variants=1, reference_images={"lighting": [Path("a.jpg"), Path("b.jpg")],
                                                                 "camera": [Path("c.jpg")]},
                               disabled_scene_axes=["camera"])
    plan = make_plan(s, 10, (640, 480))
    assert plan.stages["references"].count == 2          # photos on unchecked axes are not counted
    assert make_plan(BackgroundSwapSettings(), 10, (640, 480)).stages["references"].total_s == 0
