import os

import pytest
from PIL import Image

from gui import theme
from gui.rounded import PANEL_STYLE, _check_box, install, round_corners, toggle_image


def _has_color(img, hex_color):
    target = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return any(pixel[:3] == target and pixel[3] == 255 for pixel in img.getdata())


def test_check_box_draws_the_mark_only_when_on():
    on = _check_box(theme.ACCENT, theme.ACCENT, "#ffffff")
    off = _check_box(theme.ENTRY, theme.FG_DIM)
    assert on.size == off.size == (16, 16)
    assert _has_color(on, "#ffffff")
    assert not _has_color(off, "#ffffff")
    assert _has_color(off, theme.ENTRY)  # hollow inside, border only
    assert on.getpixel((0, 0))[3] < 16 and off.getpixel((0, 0))[3] < 16  # corners are effectively transparent


def test_round_corners_keeps_size_and_center_and_fills_corners():
    img = Image.new("RGB", (60, 40), (10, 200, 30))
    out = round_corners(img, radius=8, corner_bg="#000000")
    assert out.size == (60, 40)
    assert out.getpixel((30, 20)) == (10, 200, 30)
    assert out.getpixel((0, 0)) == (0, 0, 0) and out.getpixel((59, 39)) == (0, 0, 0)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display")
def test_install_is_idempotent_and_registers_styles():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.withdraw()
    try:
        install(root)
        first = root._rounded_images
        install(root)
        assert root._rounded_images is first
        ttk.Label(root, style=PANEL_STYLE, text="x").pack()
        ttk.Button(root, text="b", style="ToggleOff.TButton").pack()
        ttk.Entry(root).pack()
        ttk.Spinbox(root, from_=0, to=1).pack()
        ttk.Combobox(root, values=["a"]).pack()
        ttk.Progressbar(root).pack()
        ttk.Scale(root).pack()
        ttk.Scrollbar(root).pack()
        ttk.Treeview(root).pack()
        root.update()
    finally:
        root.destroy()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display")
def test_check_and_radio_use_the_box_indicator():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.withdraw()
    try:
        install(root)
        for widget_style in ("TCheckbutton", "TRadiobutton"):
            assert "RCheck.indicator" in str(ttk.Style(root).layout(widget_style))
        var = tk.BooleanVar(value=True)
        check = ttk.Checkbutton(root, text="x", variable=var)
        check.pack()
        root.update()
        assert check.instate(["selected"])
        var.set(False)
        root.update()
        assert not check.instate(["selected"])
    finally:
        root.destroy()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display")
def test_aug_toggle_shows_a_box_image_for_each_state():
    import tkinter as tk
    from tkinter import ttk

    from gui.augment_tab import _AugToggle

    root = tk.Tk()
    root.withdraw()
    try:
        install(root)
        var = tk.BooleanVar(value=False)
        toggle = _AugToggle(ttk.Frame(root), "Flip", var, lambda: None)
        assert str(toggle._btn.cget("text")) == "Flip"
        assert str(toggle._btn.cget("style")) == "ToggleOff.TButton"
        assert str(toggle._btn.cget("image")[0]) == str(toggle_image(root, False))
        toggle._click()
        assert var.get() is True
        assert str(toggle._btn.cget("text")) == "Flip"
        assert str(toggle._btn.cget("style")) == "ToggleOn.TButton"
        assert str(toggle._btn.cget("image")[0]) == str(toggle_image(root, True))
    finally:
        root.destroy()
