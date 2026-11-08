import os

import numpy as np
import pytest

from background_swap.settings import BackgroundSwapSettings, KeepItem
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
    # 부모가 파일 → 예외 없이 통과
    save_ui_state(BackgroundSwapSettings(), tmp_path / "file.txt" / "x.json")


def test_parse_axis_values():
    assert parse_axis_values(" gym ; hall\n\n arena ;") == ["gym", "hall", "arena"]
    assert format_axis_values(["gym", "hall"]) == "gym; hall"


def test_quick_mask_preview_shape_and_colors():
    img = np.full((60, 80, 3), 128, np.uint8)
    out = quick_mask_preview(img, [BBox(0, 0.5, 0.5, 0.25, 0.25)], BackgroundSwapSettings())
    assert out.shape == img.shape and tuple(out[30, 40]) != (128, 128, 128)


def test_quick_mask_preview_color_keep_only_and_text_ignored():
    img = np.full((60, 80, 3), 128, np.uint8)
    img[:20, :30] = (0, 255, 0)  # 초록 영역
    green = KeepItem(name="g", method="color", hsv_lower=[50, 100, 100],
                     hsv_upper=[70, 255, 255], min_area_ratio=0.0)
    text = KeepItem(name="t", method="text", prompt="banner")
    out = quick_mask_preview(img, [], BackgroundSwapSettings(keep_items=[green, text]))
    assert tuple(out[10, 15]) != (0, 255, 0)          # 오버레이로 색이 바뀜
    assert tuple(out[50, 70]) == (128, 128, 128)       # 나머지는 그대로


# ── Tk 스모크 ────────────────────────────────────────────────────────

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
            num_variants=5, max_attempts=2, seed=42, use_llm=False, use_vlm=False,
            keep_items=[KeepItem(name="g", method="color", hsv_lower=[50, 100, 100],
                                 hsv_upper=[70, 255, 255], required=True)],
            scene_axes={"venue": ["a", "b"], "mood": ["x"]},
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
                     "use_llm", "use_vlm", "keep_items", "scene_axes",
                     "small_box_px", "dilate_px", "feather_px"):
            assert getattr(out, name) == getattr(s, name), name
        # 비숫자 텍스트는 이전 유효값 유지 (예외 없음)
        tab._vars["num_variants"].set("abc")
        assert tab.build_settings().num_variants == 5
        tab.destroy()
        # <Destroy> 시 UI 상태가 저장된다
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
        from gui.augment_tab import _PREVIEW_SIZE
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
        # 이미지 60x80 (h x w): 미리보기 (190, 140) → (95/..): x=int(190*80/380)=40, y=int(140*60/280)=30
        tab._on_preview_click(SimpleNamespace(widget=lbl, x=ox + 190, y=oy + 140))
        assert got == [(40, 30)] and done == [1]
        # 1회성: 두 번째 클릭은 무시
        tab._on_preview_click(SimpleNamespace(widget=lbl, x=ox + 1, y=oy + 1))
        assert got == [(40, 30)]
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
        from gui.background_tab import BackgroundTab, KeepItemDialog
        tab = BackgroundTab(root)
        dlg = KeepItemDialog(tab, None, ["dup"])
        dlg._ok()                                   # 이름 없음
        dlg._name_var.set("dup"); dlg._ok()         # 중복
        dlg._name_var.set("g"); dlg._ok()           # color 인데 색 미선택
        assert dlg.result is None and len(errors) == 3
        dlg._lower, dlg._upper = [50, 100, 100], [70, 255, 255]
        dlg._ok()
        assert dlg.result == KeepItem(name="g", method="color", hsv_lower=[50, 100, 100],
                                      hsv_upper=[70, 255, 255])
        dlg2 = KeepItemDialog(tab, None, [])
        dlg2._name_var.set("t"); dlg2._method.set("text"); dlg2._ok()   # 프롬프트 없음
        assert dlg2.result is None and len(errors) == 4
        dlg2._prompt.set("blue banner"); dlg2._ok()
        assert dlg2.result == KeepItem(name="t", method="text", prompt="blue banner")
        tab.destroy()
    finally:
        root.destroy()
