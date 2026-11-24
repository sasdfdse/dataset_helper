import json

import pytest

from background_swap.common import hardware
from background_swap.common.estimate import EtaTracker, make_plan, speed_key
from background_swap.common.hardware import (HardwareProfile, load_plan, load_profile, merge_probe,
                                             record_speed, save_profile)
from background_swap.common.settings import BackgroundSwapSettings


@pytest.mark.parametrize("vram, expected", [
    (0.0, ("4bit", "4bit", "sequential")),       # no CUDA / very small GPU
    (5.8, ("4bit", "4bit", "sequential")),
    (7.62, ("fp16", "4bit", "model")),           # 8GB laptop (reference PC: must match the old fixed values)
    (11.6, ("fp16", "fp16", "model")),           # 12GB
    (23.5, ("fp16", "fp16", "none")),            # 24GB
])
def test_load_plan_by_vram(vram, expected):
    p = load_plan(vram)
    assert (p.llm_load, p.vlm_load, p.inpaint_offload) == expected


def test_base_factor_from_probe_speed():
    ref = hardware.REFERENCE_TFLOPS
    assert HardwareProfile("ref", 7.6, ref).base_factor == pytest.approx(1.0)
    assert HardwareProfile("fast", 24, ref * 2).base_factor == pytest.approx(0.5)
    assert HardwareProfile("slow", 4, ref / 3).base_factor == pytest.approx(3.0)
    assert HardwareProfile().base_factor == 1.0 and not HardwareProfile().known


def test_profile_round_trip_keeps_measured_per_gpu(tmp_path):
    path = tmp_path / "hw.json"
    a = HardwareProfile("GPU A", 8.0, 20.0, {"inpaint": 1.2})
    save_profile(a, path)
    save_profile(HardwareProfile("GPU B", 24.0, 80.0), path)
    assert load_profile(path).gpu_name == "GPU B"              # last written GPU
    assert load_profile(path, "GPU A") == a
    # re-probing keeps the measured ratios for the same GPU
    merged = merge_probe(path, "GPU A", 8.0, 22.0)
    assert merged.tflops == 22.0 and merged.measured == {"inpaint": 1.2}
    assert merge_probe(path, "GPU C", 4.0, 5.0).measured == {}


def test_broken_profile_file_is_ignored(tmp_path):
    path = tmp_path / "hw.json"
    path.write_text("{oops", encoding="utf-8")
    assert not load_profile(path).known
    path.write_text(json.dumps({"last_gpu": "X", "gpus": {"X": {"measured": {"inpaint": "fast", "masks": 2}}}}),
                    encoding="utf-8")
    assert load_profile(path).measured == {"masks": 2.0}
    save_profile(HardwareProfile("Y", 1, 1), path)          # saving works despite a corrupt entry
    assert load_profile(path, "Y").gpu_name == "Y"


def test_record_speed_smooths_and_clips():
    p = HardwareProfile("G", 8, 20)
    p = record_speed(p, {"inpaint": 2.0, "masks": 0.0, "prompts": 1000.0})
    assert p.measured == {"inpaint": 2.0, "prompts": 20.0}
    p = record_speed(p, {"inpaint": 1.0})
    assert p.measured["inpaint"] == pytest.approx(1.5)
    assert p.factor("inpaint") == pytest.approx(1.5) and p.factor("validate") == p.base_factor


def test_plan_uses_profile_factors():
    s = BackgroundSwapSettings(num_variants=1)
    base = make_plan(s, 10, (640, 480))
    slow = make_plan(s, 10, (640, 480), HardwareProfile("G", 8, 20, {speed_key(s, "inpaint"): 2.0}))
    assert slow.stages["inpaint"].per_item_s == pytest.approx(2 * base.stages["inpaint"].per_item_s)
    assert slow.stages["masks"].per_item_s == pytest.approx(
        base.stages["masks"].per_item_s * slow_base_factor())


def slow_base_factor():
    return HardwareProfile("G", 8, 20).base_factor


def test_tracker_speed_ratios_need_two_samples():
    s = BackgroundSwapSettings(num_variants=1, use_llm=False)
    base = make_plan(s, 3, (640, 480))
    eta = EtaTracker(base)
    per = base.stages["inpaint"].per_item_s
    eta.update("inpaint", 0, 3, now=0.0)
    eta.update("inpaint", 1, 3, now=30.0)
    eta.update("inpaint", 2, 3, now=30.0 + 2 * per)
    assert eta.speed_ratios(base) == {}                     # a single sample is not saved
    eta.update("inpaint", 3, 3, now=30.0 + 4 * per)
    assert eta.speed_ratios(base) == {"inpaint": pytest.approx(2.0)}
    assert "prompts" not in eta.speed_ratios(base)          # LLM off: stages with a 0s baseline are skipped


def test_small_install_without_bitsandbytes_uses_fp16():
    p = load_plan(5.8, can_4bit=False)
    assert (p.llm_load, p.vlm_load, p.inpaint_offload) == ("fp16", "fp16", "sequential")


def test_small_preset_estimate_uses_sd15_timing_and_separate_speed_keys():
    import dataclasses
    from background_swap.common.estimate import inpaint_seconds
    from background_swap.common.settings import MODEL_PRESETS
    large = BackgroundSwapSettings()
    small = dataclasses.replace(large, models=MODEL_PRESETS["small"], use_vlm=False)
    # 640x480: Small draws at source size (~6s), Large upscales to 1024x768 (~20s)
    assert inpaint_seconds(640, 480, small) == pytest.approx(6.3, abs=0.6)
    assert inpaint_seconds(640, 480, large) > 3 * inpaint_seconds(640, 480, small)
    assert speed_key(small, "inpaint") != speed_key(large, "inpaint")
    fast = dataclasses.replace(large, fast_mode=True)
    assert speed_key(fast, "inpaint") != speed_key(large, "inpaint")
