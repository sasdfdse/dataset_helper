from background_swap.common.settings import (
    BackgroundSwapSettings, KeepItem, active_scene_axes, axis_label, effective_settings,
    generation_diff, load_settings, save_settings, settings_from_dict, validate_settings,
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
    assert len(errs) == 3   # images_dir, labels_dir, output_dir missing


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


def _axes_settings(**kw):
    return BackgroundSwapSettings(scene_axes={"venue": ["gym"], "lighting": ["dim"], "camera": []}, **kw)


def test_disabled_axes_roundtrip_and_default(tmp_path):
    assert BackgroundSwapSettings().disabled_scene_axes == []
    s = _axes_settings(disabled_scene_axes=["camera"])
    save_settings(s, tmp_path / "s.json")
    assert load_settings(tmp_path / "s.json") == s
    assert settings_from_dict({"disabled_scene_axes": ["lighting", 3]}).disabled_scene_axes == ["lighting", "3"]


def test_active_scene_axes_and_effective_settings():
    s = _axes_settings(disabled_scene_axes=["lighting", "camera"])
    assert active_scene_axes(s) == {"venue": ["gym"]}
    e = effective_settings(s)
    assert e.scene_axes == {"venue": ["gym"]} and e.disabled_scene_axes == []
    assert s.scene_axes["lighting"] == ["dim"]          # the original is unchanged


def test_generation_diff_only_looks_at_checked_axes():
    a = _axes_settings(disabled_scene_axes=["camera"])
    b = BackgroundSwapSettings(scene_axes={"venue": ["gym"], "lighting": ["dim"], "camera": ["blur"]},
                               disabled_scene_axes=["camera"])
    assert generation_diff(a, b) == []                    # only an unchecked row's values changed
    c = _axes_settings(disabled_scene_axes=["camera", "lighting"])
    assert generation_diff(a, c) == ["scene_axes"]        # a used row changed


def test_validate_scene_axes_only_checks_checked_rows(tmp_path):
    (tmp_path / "i").mkdir()
    (tmp_path / "l").mkdir()
    base = dict(images_dir=tmp_path / "i", labels_dir=tmp_path / "l", output_dir=tmp_path / "o")
    assert validate_settings(_axes_settings(disabled_scene_axes=["camera"], **base)) == []
    errs = validate_settings(_axes_settings(**base))
    assert any("사진 느낌" in e for e in errs)            # checked row left empty
    errs = validate_settings(_axes_settings(disabled_scene_axes=["venue", "lighting", "camera"], **base))
    assert any("배경 조합" in e and "1개 이상" in e for e in errs)


def test_axis_label():
    assert [axis_label(n) for n in ("venue", "lighting", "surroundings_floor", "context", "camera")] == [
        "장소", "조명", "바닥", "주변 풍경", "사진 느낌"]
    assert axis_label("mood") == "mood"


def test_model_presets_and_fast_mode_needs_sdxl(tmp_path):
    import dataclasses
    from background_swap.common.settings import MODEL_PRESETS, ModelIds, is_sdxl, model_preset
    assert model_preset(ModelIds()) == "large"
    assert model_preset(MODEL_PRESETS["small"]) == "small"
    assert model_preset(dataclasses.replace(ModelIds(), llm="me/my-llm")) == "custom"
    assert is_sdxl(MODEL_PRESETS["large"].inpainter) and not is_sdxl(MODEL_PRESETS["small"].inpainter)
    base = dict(images_dir=tmp_path, labels_dir=tmp_path, output_dir=tmp_path / "out")
    assert validate_settings(BackgroundSwapSettings(fast_mode=True, **base)) == []
    errs = validate_settings(BackgroundSwapSettings(fast_mode=True, models=MODEL_PRESETS["small"], **base))
    assert len(errs) == 1 and "빠른 모드" in errs[0]
