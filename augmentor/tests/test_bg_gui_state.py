import os
from pathlib import Path

import numpy as np
import pytest

from background_swap.common.settings import BackgroundSwapSettings, KeepItem
from core.label import BBox
from gui.background_state import (format_axis_values, load_ui_state,
                                  parse_axis_values, quick_mask_preview,
                                  save_ui_state)


def test_ui_state_roundtrip_and_corrupt(tmp_path):
    p = tmp_path / "ui.json"
    s = BackgroundSwapSettings(num_variants=7)
    save_ui_state(s, p)
    assert load_ui_state(p).num_variants == 7
    p.write_text("{broken")
    assert load_ui_state(p) == BackgroundSwapSettings()


def test_load_ui_state_missing_returns_default(tmp_path):
    assert load_ui_state(tmp_path / "nope.json") == BackgroundSwapSettings()


def test_save_ui_state_swallows_errors(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    # parent is a file -> passes without raising
    save_ui_state(BackgroundSwapSettings(), tmp_path / "file.txt" / "x.json")


def test_parse_axis_values():
    assert parse_axis_values(" gym ; hall\n\n arena ;") == ["gym", "hall", "arena"]
    assert format_axis_values(["gym", "hall"]) == "gym; hall"
    assert format_axis_values([1, 2]) == "1; 2"


def test_quick_mask_preview_shape_and_colors():
    img = np.full((60, 80, 3), 128, np.uint8)
    out = quick_mask_preview(img, [BBox(0, 0.5, 0.5, 0.25, 0.25)], BackgroundSwapSettings())
    assert out.shape == img.shape and tuple(out[30, 40]) != (128, 128, 128)


def test_quick_mask_preview_color_keep_only_and_text_ignored():
    img = np.full((60, 80, 3), 128, np.uint8)
    img[:20, :30] = (0, 255, 0)
    green = KeepItem(name="g", method="color", hsv_lower=[50, 100, 100],
                     hsv_upper=[70, 255, 255], min_area_ratio=0.0)
    text = KeepItem(name="t", method="text", prompt="banner")
    out = quick_mask_preview(img, [], BackgroundSwapSettings(keep_items=[green, text]))
    assert tuple(out[10, 15]) != (0, 255, 0)          # overlay changes the color
    assert tuple(out[50, 70]) == (128, 128, 128)       # the rest is unchanged


needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no DISPLAY")


def _make_dataset(root):
    import cv2
    images, labels = root / "images", root / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(2):
        cv2.imwrite(str(images / f"img{i}.jpg"), np.full((60, 80, 3), 100 + i, np.uint8))
        (labels / f"img{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    (labels / "classes.txt").write_text("robot\n")
    return images, labels


@needs_display
def test_tab_apply_preview_build_roundtrip(tmp_path, monkeypatch):
    state = tmp_path / "ui_state.json"
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", state)
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        images, labels = _make_dataset(tmp_path)
        tab = BackgroundTab(root)
        tab.pack()
        s = BackgroundSwapSettings(
            images_dir=images, labels_dir=labels, output_dir=tmp_path / "out",
            class_names_file=labels / "classes.txt", frame_stride=1,
            num_variants=5, max_attempts=2, seed=42, use_llm=False, use_vlm=False, fast_mode=True,
            keep_items=[KeepItem(name="g", method="color", hsv_lower=[50, 100, 100],
                                 hsv_upper=[70, 255, 255], required=True)],
            scene_axes={"venue": ["a", "b"], "mood": ["x"]}, disabled_scene_axes=["mood"],
            small_box_px=16, dilate_px=3, feather_px=2,
        )
        tab.apply_settings(s)
        tab.update()
        tab.refresh_preview_now()
        assert tab._mask_lbl.image is not None
        assert tab._orig_lbl.image is not None
        out = tab.build_settings()
        for name in ("images_dir", "labels_dir", "output_dir", "class_names_file",
                     "frame_stride", "num_variants", "max_attempts", "seed",
                     "use_llm", "use_vlm", "fast_mode", "keep_items", "scene_axes", "disabled_scene_axes",
                     "small_box_px", "dilate_px", "feather_px"):
            assert getattr(out, name) == getattr(s, name), name
        # non-numeric text keeps the last valid value (no exception)
        tab._vars["num_variants"].set("abc")
        assert tab.build_settings().num_variants == 5
        tab.destroy()
        # UI state is saved on <Destroy>
        assert load_ui_state(state).num_variants == 5
    finally:
        root.destroy()


@needs_display
def test_app_launch_smoke(tmp_path, monkeypatch):
    import tkinter as tk
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    from gui.app import App
    try:
        app = App()
    except tk.TclError:
        pytest.skip("cannot open display")
    try:
        app.withdraw()
        nb = next(w for w in app.winfo_children() if w.winfo_class() == "TNotebook")
        nb.select(nb.tabs()[-1])
        for _ in range(3):
            app.update()
    finally:
        app.destroy()


@needs_display
def test_click_to_pick_maps_to_original_coords(tmp_path, monkeypatch):
    import tkinter as tk
    from types import SimpleNamespace
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import _PREVIEW_SIZE
        from gui.background_tab import BackgroundTab
        images, labels = _make_dataset(tmp_path)
        tab = BackgroundTab(root)
        tab.pack()
        tab.apply_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels))
        tab.refresh_preview_now()
        tab.update()
        got, done = [], []
        tab.begin_pick(lambda img, x, y: got.append((x, y)), lambda: done.append(1))
        lbl = tab._orig_lbl
        pw, ph = _PREVIEW_SIZE
        ox, oy = max(0, (lbl.winfo_width() - pw) // 2), max(0, (lbl.winfo_height() - ph) // 2)
        # image 60x80 (h x w): clicking the panel center maps to source coordinates
        cx, cy = pw // 2, ph // 2
        expected = (int(cx * 80 / pw), int(cy * 60 / ph))
        tab._on_preview_click(SimpleNamespace(widget=lbl, x=ox + cx, y=oy + cy))
        assert got == [expected] and done == [1]
        # one-shot: the second click is ignored
        tab._on_preview_click(SimpleNamespace(widget=lbl, x=ox + 1, y=oy + 1))
        assert got == [expected]
        tab.destroy()
    finally:
        root.destroy()


@needs_display
def test_keep_item_dialog_validation_and_result(tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import messagebox
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: errors.append(a))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        from gui.keep_item_dialog import KeepItemDialog
        tab = BackgroundTab(root)
        dlg = KeepItemDialog(tab, None, ["dup"])
        dlg._ok()                                   # no name
        dlg._name_var.set("dup"); dlg._ok()         # duplicate
        dlg._name_var.set("g"); dlg._ok()           # color method without a picked color
        assert dlg.result is None and len(errors) == 3
        dlg._lower, dlg._upper = [50, 100, 100], [70, 255, 255]
        dlg._ok()
        assert dlg.result == KeepItem(name="g", method="color", hsv_lower=[50, 100, 100],
                                      hsv_upper=[70, 255, 255])
        dlg2 = KeepItemDialog(tab, None, [])
        dlg2._name_var.set("t"); dlg2._method.set("text"); dlg2._ok()   # no prompt
        assert dlg2.result is None and len(errors) == 4
        dlg2._prompt.set("blue banner"); dlg2._ok()
        assert dlg2.result == KeepItem(name="t", method="text", prompt="blue banner")
        tab.destroy()
    finally:
        root.destroy()


@needs_display
def test_tab_survives_bad_typed_ui_state(tmp_path, monkeypatch):
    import json
    import time
    import tkinter as tk
    from background_swap.common.settings import DEFAULT_SCENE_AXES
    state = tmp_path / "ui_state.json"
    state.write_text(json.dumps({"scene_axes": {"venue": [1, 2]}, "keep_items": [{"name": "x", "method": "color"}],
                                 "num_variants": "many"}))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        # 1) file with wrongly typed values: builds without raising, values coerced to strings
        tab = BackgroundTab(root, state_path=state)
        tab.update()
        assert tab.build_settings().scene_axes == {"venue": ["1", "2"]}
        tab.destroy()
        # 2) apply_settings itself fails: fall back to defaults and overwrite the file with a valid state
        monkeypatch.setattr("gui.background_tab.load_ui_state",
                            lambda p: BackgroundSwapSettings(scene_axes={"venue": None}))
        tab = BackgroundTab(root, state_path=state)
        # the notice must survive the scheduled preview refresh (150ms)
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            tab.update()
            time.sleep(0.01)
        assert tab.build_settings().scene_axes == DEFAULT_SCENE_AXES
        assert "기본값" in tab._status_lbl.cget("text")
        tab.destroy()
        assert load_ui_state(state).scene_axes == DEFAULT_SCENE_AXES
    finally:
        root.destroy()


@needs_display
def test_scene_axes_are_checkbox_options(tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk
    from background_swap.common.settings import DEFAULT_SCENE_AXES
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        tab = BackgroundTab(root, state_path=tmp_path / "ui_state.json")
        tab.apply_settings(BackgroundSwapSettings(disabled_scene_axes=["camera"]))
        tab.update()

        def walk(w):
            yield w
            for c in w.winfo_children():
                yield from walk(c)
        texts = [str(w.cget("text")) for w in walk(tab) if isinstance(w, (ttk.Button, ttk.Labelframe))]
        assert not any("축 추가" in t or "축 삭제" in t for t in texts)
        assert any(t.startswith("배경 조합") for t in texts)

        rows = {row.name: row for row in tab._axis_rows}
        assert list(rows) == list(DEFAULT_SCENE_AXES)
        assert [str(r.check.cget("text")) for r in rows.values()] == [
            "장소", "조명", "바닥", "주변 풍경", "사진 느낌"]
        assert not rows["camera"].enabled.get() and rows["camera"].entry.instate(["disabled"])
        assert rows["venue"].enabled.get() and not rows["venue"].entry.instate(["disabled"])

        rows["lighting"].check.invoke()            # turn lighting off
        rows["camera"].check.invoke()              # turn camera on
        out = tab.build_settings()
        assert out.disabled_scene_axes == ["lighting"]
        assert out.scene_axes == DEFAULT_SCENE_AXES  # values of unchecked rows are kept
        assert rows["lighting"].entry.instate(["disabled"]) and not rows["camera"].entry.instate(["disabled"])
        tab.destroy()
    finally:
        root.destroy()


@needs_display
def test_scene_axis_examples_are_placeholders(tmp_path, monkeypatch):
    import tkinter as tk
    from background_swap.common.settings import DEFAULT_SCENE_AXES
    from gui.background_state import format_axis_values
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        tab = BackgroundTab(root, state_path=tmp_path / "ui_state.json")
        tab.apply_settings(BackgroundSwapSettings(
            scene_axes={**DEFAULT_SCENE_AXES, "camera": ["blurry photo"]}))
        tab.update()
        rows = {row.name: row for row in tab._axis_rows}
        venue, camera = rows["venue"], rows["camera"]

        # example values show only as grey hint text, not as real text
        assert venue.values.get() == "" and venue.placeholder.visible
        assert venue.placeholder.label.cget("text") == format_axis_values(DEFAULT_SCENE_AXES["venue"])
        # typed values stay as real text and the hint hides
        assert camera.values.get() == "blurry photo" and not camera.placeholder.visible

        # an empty field uses the example values (settings unchanged)
        assert tab.build_settings().scene_axes == {**DEFAULT_SCENE_AXES, "camera": ["blurry photo"]}

        # clicking (focus) hides the hint; leaving the field empty shows it again
        # (a hidden window has no real keyboard focus, so send focus events directly)
        focused = []
        monkeypatch.setattr(venue.entry, "focus_set", lambda: focused.append(1))
        venue.placeholder.label.event_generate("<Button-1>")
        assert focused == [1]                      # clicking the hint still focuses the field
        venue.entry.event_generate("<FocusIn>")
        tab.update()
        assert not venue.placeholder.visible
        venue.entry.event_generate("<FocusOut>")
        tab.update()
        assert venue.placeholder.visible
        # typed input is used
        venue.values.set("hotel ballroom")
        tab.update()
        assert not venue.placeholder.visible
        assert tab.build_settings().scene_axes["venue"] == ["hotel ballroom"]
        tab.destroy()
    finally:
        root.destroy()


@needs_display
def test_model_size_preset_switches_models(tmp_path, monkeypatch):
    import tkinter as tk
    from background_swap.common.settings import MODEL_PRESETS
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        tab = BackgroundTab(root, state_path=tmp_path / "ui_state.json")
        tab.apply_settings(BackgroundSwapSettings(fast_mode=True))
        tab.update()
        assert tab._preset_var.get() == "large" and not tab._fast_check.instate(["disabled"])
        tab._apply_preset("small")
        s = tab.build_settings()
        assert s.models == MODEL_PRESETS["small"] and not s.use_vlm and not s.fast_mode
        assert tab._preset_var.get() == "small" and tab._fast_check.instate(["disabled"])
        assert "Small" in tab._preset_note.cget("text")
        tab._vars["model_llm"].set("me/custom-llm")          # a manual change matches neither preset
        assert tab._preset_var.get() == "" and "직접" in tab._preset_note.cget("text")
        tab._apply_preset("large")
        s = tab.build_settings()
        assert s.models == MODEL_PRESETS["large"] and s.use_vlm
        tab.destroy()
    finally:
        root.destroy()


@needs_display
def test_axis_rows_take_optional_reference_photos(tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import filedialog
    from background_swap.common.settings import DEFAULT_SCENE_AXES
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    photos = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **k: photos)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        from gui.background_tab import BackgroundTab
        tab = BackgroundTab(root, state_path=tmp_path / "ui_state.json")
        tab.apply_settings(BackgroundSwapSettings())
        tab.update()
        rows = {row.name: row for row in tab._axis_rows}
        # reference photos are optional: without them nothing changes
        assert all(not r.refs and "사진" in str(r.ref_btn.cget("text")) for r in rows.values())
        assert tab.build_settings().reference_images == {}
        light = rows["lighting"]
        light.ref_btn.invoke()
        assert str(light.ref_btn.cget("text")) == "📷 2장" and light.ref_clear.winfo_manager()
        assert "참고 사진" in light.placeholder.label.cget("text")
        s = tab.build_settings()
        assert s.reference_images == {"lighting": [Path(p) for p in photos]}
        assert s.scene_axes["lighting"] == []                         # empty field + photo = photo captions only
        assert s.scene_axes["venue"] == DEFAULT_SCENE_AXES["venue"]   # other rows unchanged
        light.values.set("warm light")
        assert tab.build_settings().scene_axes["lighting"] == ["warm light"]
        # survives reloading
        tab.apply_settings(s)
        light = {row.name: row for row in tab._axis_rows}["lighting"]
        assert light.refs == [Path(p) for p in photos] and light.values.get() == ""
        light.ref_clear.invoke()
        assert tab.build_settings().reference_images == {} and not light.ref_clear.winfo_manager()
        assert tab.build_settings().scene_axes["lighting"] == DEFAULT_SCENE_AXES["lighting"]
        tab.destroy()
    finally:
        root.destroy()
