"""Linux 파일/폴더 선택 창 목록이 세로 한 열 + 휠 스크롤인지 확인한다."""
import tkinter as tk

import pytest

from gui import vertical_filedialog


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("디스플레이 없음")
    if r.tk.call("tk", "windowingsystem") != "x11":
        r.destroy()
        pytest.skip("X11 전용")
    r.geometry("400x200")
    yield r
    r.destroy()


def _icon_list(root):
    vertical_filedialog.install(root)
    icons = root.tk.call("::tk::IconList", ".icons")
    root.tk.call("pack", icons, "-fill", "both", "-expand", "1")
    image = root.tk.call("image", "create", "photo", "-width", "16", "-height", "16")
    root.tk.call(icons, "add", image, [f"folder_{i:02d}" for i in range(60)])
    root.update()       # Arrange runs at idle
    return ".icons.cHull.canvas", ".icons.cHull.sbar"


def test_items_in_one_vertical_column(root):
    canvas, sbar = _icon_list(root)
    assert str(root.tk.call(sbar, "cget", "-orient")) == "vertical"
    rects = root.tk.splitlist(root.tk.call(canvas, "find", "withtag", "rect"))
    xs = {root.tk.splitlist(root.tk.call(canvas, "coords", r))[0] for r in rects}
    assert len(rects) == 60 and len(xs) == 1


def test_mouse_wheel_scrolls_up_and_down(root):
    canvas, _ = _icon_list(root)
    top = lambda: float(root.tk.splitlist(root.tk.call(canvas, "yview"))[0])
    assert top() == 0.0
    root.tk.call("event", "generate", canvas, "<Button-5>")
    root.update()
    down = top()
    assert down > 0.0
    root.tk.call("event", "generate", canvas, "<Button-4>")
    root.update()
    assert top() < down
