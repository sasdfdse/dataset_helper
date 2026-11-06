from background_swap.settings import (
    BackgroundSwapSettings, KeepItem, generation_diff, load_settings,
    save_settings, settings_from_dict, validate_settings,
)


def test_roundtrip_preserves_all_fields(tmp_path):
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"out",
        keep_items=[KeepItem("field", "color", required=True, hsv_lower=[35, 40, 40], hsv_upper=[85, 255, 255]),
                    KeepItem("fence", "text", prompt="wooden arena fence")])
    save_settings(s, tmp_path/"run_settings.json")
    assert load_settings(tmp_path/"run_settings.json") == s


def test_from_dict_fills_missing_with_defaults():
    s = settings_from_dict({"num_variants": 5})
    assert s.num_variants == 5 and s.max_attempts == 3 and s.dilate_px == 6
    assert list(s.scene_axes) == ["venue", "lighting", "surroundings_floor", "context", "camera"]
    assert s.models.inpainter == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def test_from_dict_ignores_unknown_keys():
    s = settings_from_dict({"seed": 4, "future_field": 1, "models": {"vlm": "x", "other": 2},
                            "keep_items": [{"name": "a", "method": "text", "prompt": "p", "zzz": 1}]})
    assert s.seed == 4 and s.models.vlm == "x" and s.keep_items[0].prompt == "p"


def test_defaults_are_not_shared_between_instances():
    a, b = BackgroundSwapSettings(), BackgroundSwapSettings()
    a.scene_axes["venue"].append("x")
    a.keep_items.append(KeepItem("k", "text", prompt="p"))
    assert "x" not in b.scene_axes["venue"] and b.keep_items == []


def test_generation_diff_ignores_output_dir(tmp_path):
    a = BackgroundSwapSettings(output_dir=tmp_path/"a")
    b = BackgroundSwapSettings(output_dir=tmp_path/"b", seed=7)
    assert generation_diff(a, b) == ["seed"]


def test_generation_diff_compares_nested_by_value():
    a, b = BackgroundSwapSettings(), BackgroundSwapSettings()
    assert generation_diff(a, b) == []
    b.models.llm = "other"
    b.keep_items.append(KeepItem("k", "text", prompt="p"))
    assert generation_diff(a, b) == ["keep_items", "models"]


def test_validate_settings_requires_dirs():
    errs = validate_settings(BackgroundSwapSettings())
    assert len(errs) == 3   # images_dir, labels_dir, output_dir 미지정


def test_validate_settings_rejects_output_equal_to_input(tmp_path):
    (tmp_path/"img").mkdir(); (tmp_path/"lbl").mkdir()
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"img")
    assert any("출력" in e for e in validate_settings(s))


def test_validate_settings_color_item_needs_hsv(tmp_path):
    for d in ("img", "lbl"): (tmp_path/d).mkdir()
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"out",
                               keep_items=[KeepItem("field", "color")])
    assert any("field" in e for e in validate_settings(s))


def test_validate_settings_ranges(tmp_path):
    for d in ("img", "lbl"): (tmp_path/d).mkdir()
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"out",
                               num_variants=0, max_attempts=11, frame_stride=0)
    assert len(validate_settings(s)) == 3
