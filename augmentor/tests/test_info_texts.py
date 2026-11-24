"""Info 창: 두 탭 모두 오른쪽 아래 버튼으로 항목 설명을 띄운다."""
import os
import re

import pytest

from gui.info_texts import AUGMENT_INFO, BACKGROUND_INFO

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no DISPLAY")


def test_augment_info_explains_every_control():
    import inspect
    from gui import augment_tab
    src = inspect.getsource(augment_tab.AugmentTab._build_settings)
    labels = re.findall(r'_aug_section\(frame, row, "([^"]+)"', src)
    assert len(labels) >= 15
    for label in labels + ["Multiplier", "images/ folder", "labels/ folder"]:
        assert label in AUGMENT_INFO, label
    assert "_aug1" in AUGMENT_INFO          # notes that results go to the same folder


def test_background_info_explains_every_setting():
    from gui.background_tab import _ADV_NUM, _DATA_NUM, _GEN_NUM, _MODEL_LABELS
    from background_swap.common.settings import SCENE_AXIS_LABELS
    for _, _, label, _ in _DATA_NUM + _GEN_NUM + _ADV_NUM:
        name = label.split(" (")[0]
        assert name.lower() in BACKGROUND_INFO.lower(), name
    for _, label in _MODEL_LABELS:
        assert label in BACKGROUND_INFO, label
    for label in list(SCENE_AXIS_LABELS.values()) + ["Images folder", "Labels folder", "Classes file",
                                                      "Output folder", "Use LLM", "Use VLM", "빠른 모드",
                                                      "Large", "Small", "📷 사진", "필수", "largest_only",
                                                      "convex_hull", "min_area_ratio", "H/S/V tol",
                                                      "resolution_mode", "Preview", "Run", "Stop", "Report",
                                                      "Export", "accepted", "rejected", "skipped", "exhausted"]:
        assert label in BACKGROUND_INFO, label


@needs_display
def test_augment_tab_info_button_opens_window(monkeypatch):
    import tkinter as tk
    from gui import augment_tab
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        windows = []
        monkeypatch.setattr(augment_tab, "ReportWindow",
                            lambda parent, text, title, **k: windows.append((title, text)))
        tab = augment_tab.AugmentTab(root)
        tab._info_btn.invoke()
        assert windows == [("Info — Augmentation", AUGMENT_INFO)]
        tab.destroy()
    finally:
        root.destroy()


@needs_display
def test_info_window_highlights_headings():
    import tkinter as tk
    from gui.background_dialogs import ReportWindow
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    try:
        w = ReportWindow(root, "[제목]\n내용\n[둘째]", title="t", size="300x200")
        w.withdraw()
        ranges = [str(r) for r in w.text_widget.tag_ranges("heading")]
        assert ranges == ["1.0", "1.4", "3.0", "3.4"]
    finally:
        root.destroy()
