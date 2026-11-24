"""ttk 위젯의 둥근 모서리.

ttk는 둥근 모서리를 직접 지원하지 않으므로 PIL로 그린 둥근 사각형을 이미지 요소로 등록한다.
이미지는 border(9-slice)로 늘어나므로 모서리 곡률은 위젯 크기와 무관하게 유지된다.
"""
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageTk
from tkinter import ttk

from gui import theme

RADIUS = 6
PANEL_RADIUS = 8
PANEL_STYLE = "Panel.TLabel"

_SUPERSAMPLE = 4
_CHECK_SIZE = 16
_CHECK_GAP = 6
_FIELD_ELEMENTS = {"Entry.field", "Spinbox.field", "Combobox.field"}


def _rounded(size: Tuple[int, int], radius: int, fill: str, corner_bg: str = theme.BG,
             outline: str = "", corners: Tuple[bool, bool, bool, bool] = (True, True, True, True)) -> Image.Image:
    """corner_bg 위에 둥근 사각형을 그린 RGB 이미지. corners = (좌상, 우상, 우하, 좌하)."""
    w, h = size
    s = _SUPERSAMPLE
    img = Image.new("RGB", (w * s, h * s), corner_bg)
    ImageDraw.Draw(img).rounded_rectangle(
        (0, 0, w * s - 1, h * s - 1), radius * s, fill=fill,
        outline=outline or fill, width=s, corners=corners,
    )
    return img.resize(size, Image.LANCZOS)


def _check_box(fill: str, outline: str, mark: str = "", size: int = _CHECK_SIZE) -> Image.Image:
    """둥근 체크박스 RGBA 이미지. mark 색을 주면 체크 표시를 그리고, 바깥 모서리는 투명하다."""
    s = _SUPERSAMPLE
    img = Image.new("RGBA", (size * s, size * s), (*ImageColor.getrgb(fill), 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, size * s - 1, size * s - 1), 3 * s, fill=fill, outline=outline, width=s + s // 2)
    if mark:
        points = [(x * size * s, y * size * s) for x, y in ((0.22, 0.53), (0.42, 0.72), (0.78, 0.29))]
        draw.line(points, fill=mark, width=2 * s, joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def _with_gap(img: Image.Image, gap: int = _CHECK_GAP) -> Image.Image:
    """글자와 붙지 않도록 이미지 오른쪽에 투명한 여백을 더한다."""
    out = Image.new("RGBA", (img.width + gap, img.height), (0, 0, 0, 0))
    out.paste(img, (0, 0))
    return out


def toggle_image(widget, on: bool) -> ImageTk.PhotoImage:
    """토글 버튼에 넣을 체크박스 이미지. 버튼 색 위에서 보이도록 체크박스와 색이 다르다."""
    top = widget.winfo_toplevel()
    install(top)
    return top._rounded_toggles[bool(on)]


def round_corners(img: Image.Image, radius: int = PANEL_RADIUS, corner_bg: str = theme.BG) -> Image.Image:
    """미리보기 이미지의 네 모서리를 corner_bg 색으로 둥글게 깎는다. 크기는 그대로."""
    w, h = img.size
    s = _SUPERSAMPLE
    mask = Image.new("L", (w * s, h * s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius * s, fill=255)
    mask = mask.resize(img.size, Image.LANCZOS)
    out = Image.new("RGB", img.size, corner_bg)
    out.paste(img.convert("RGB"), (0, 0), mask)
    return out


def _swap_elements(layout: Iterable, mapping: Dict[str, str]) -> list:
    """레이아웃 트리에서 요소 이름을 바꾼 사본을 돌려준다."""
    result = []
    for name, opts in layout:
        opts = dict(opts)
        if "children" in opts:
            opts["children"] = _swap_elements(opts["children"], mapping)
        result.append((mapping.get(name, name), opts))
    return result


def install(root) -> None:
    """현재 ttk 테마에 둥근 스타일을 등록한다. 같은 root에 두 번 호출해도 안전하다."""
    if getattr(root, "_rounded_images", None) is not None:
        return
    style = ttk.Style(root)
    images = []

    def keep(img: Image.Image) -> ImageTk.PhotoImage:
        image = ImageTk.PhotoImage(img, master=root)
        images.append(image)
        return image

    def photo(*args, **kwargs) -> ImageTk.PhotoImage:
        return keep(_rounded(*args, **kwargs))

    def element(name, normal, *states, border=RADIUS + 2, padding=0, sticky="nsew"):
        style.element_create(name, "image", normal, *states, border=border, padding=padding, sticky=sticky)

    box = (32, 32)

    element("RButton.bg", photo(box, RADIUS, theme.ACCENT),
            ("disabled", photo(box, RADIUS, theme.SEL)),
            ("pressed", photo(box, RADIUS, theme.ACCENT2)),
            ("active", photo(box, RADIUS, theme.ACCENT2)))
    element("RToggleOff.bg", photo(box, RADIUS, theme.ENTRY),
            ("active", photo(box, RADIUS, theme.SEL)))
    button_body = [("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]
    style.layout("TButton", [("RButton.bg", {"sticky": "nsew", "children": button_body})])
    style.layout("ToggleOn.TButton", [("RButton.bg", {"sticky": "nsew", "children": button_body})])
    style.layout("ToggleOff.TButton", [("RToggleOff.bg", {"sticky": "nsew", "children": button_body})])

    element("RField.field", photo(box, RADIUS, theme.ENTRY, outline=theme.SEL),
            ("disabled", photo(box, RADIUS, theme.PANEL, outline=theme.SEL)),
            ("focus", photo(box, RADIUS, theme.ENTRY, outline=theme.ACCENT)), padding=2)
    for widget_style in ("TEntry", "TSpinbox", "TCombobox"):
        mapping = {name: "RField.field" for name in _FIELD_ELEMENTS}
        style.layout(widget_style, _swap_elements(style.layout(widget_style), mapping))

    element("RFrame.border", photo(box, PANEL_RADIUS, theme.BG, outline=theme.SEL), border=PANEL_RADIUS + 2, padding=1)
    style.layout("TLabelframe", [("RFrame.border", {"sticky": "nsew"})])
    element("RPanel.border", photo(box, PANEL_RADIUS, theme.ENTRY), border=PANEL_RADIUS + 2)
    style.layout(PANEL_STYLE, [("RPanel.border", {"sticky": "nsew", "children": [
        ("Label.padding", {"sticky": "nsew", "children": [("Label.label", {"sticky": "nsew"})]})]})])
    style.configure(PANEL_STYLE, anchor="center", background=theme.ENTRY)

    top_only = (True, True, False, False)
    element("RTab.tab", photo(box, RADIUS, theme.ENTRY, corner_bg=theme.PANEL, corners=top_only),
            ("selected", photo(box, RADIUS, theme.BG, corner_bg=theme.PANEL, corners=top_only)))
    style.layout("TNotebook.Tab", _swap_elements(style.layout("TNotebook.Tab"), {"Notebook.tab": "RTab.tab"}))

    element("RProgress.trough", photo((14, 14), 6, theme.ENTRY), border=6)
    element("RProgress.pbar", photo((14, 14), 6, theme.ACCENT, corner_bg=theme.ENTRY), border=6)
    style.layout("Horizontal.TProgressbar", [("RProgress.trough", {"sticky": "nsew", "children": [
        ("RProgress.pbar", {"side": "left", "sticky": "ns"})]})])

    element("RScale.trough", photo((16, 6), 3, theme.SEL), border=(4, 0), sticky="ew")
    element("RScale.slider", photo((14, 14), 7, theme.ACCENT), ("active", photo((14, 14), 7, theme.ACCENT2)),
            border=0, sticky="")
    style.layout("Horizontal.TScale", [("RScale.trough", {"sticky": "ew"}),
                                       ("RScale.slider", {"side": "left", "sticky": ""})])

    style.configure("Vertical.TScrollbar", troughcolor=theme.BG, bordercolor=theme.BG,
                    lightcolor=theme.BG, darkcolor=theme.BG)
    element("RScroll.thumb", photo((10, 24), 5, theme.SEL), ("active", photo((10, 24), 5, theme.FG_DIM)), border=5)
    style.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
        ("RScroll.thumb", {"expand": "1", "sticky": "nsew"})]})])

    def check_state(*args) -> ImageTk.PhotoImage:
        return keep(_with_gap(_check_box(*args)))

    element("RCheck.indicator", check_state(theme.ENTRY, theme.FG_DIM),
            ("disabled", "selected", check_state(theme.SEL, theme.SEL, theme.FG_DIM)),
            ("disabled", check_state(theme.PANEL, theme.SEL)),
            ("selected", check_state(theme.ACCENT, theme.ACCENT, "#ffffff")),
            ("active", check_state(theme.ENTRY, theme.FG)), border=0, sticky="")
    for widget_style, indicator in (("TRadiobutton", "Radiobutton.indicator"), ("TCheckbutton", "Checkbutton.indicator")):
        style.layout(widget_style, _swap_elements(style.layout(widget_style), {indicator: "RCheck.indicator"}))
        style.map(widget_style, background=[("active", theme.BG)])

    root._rounded_toggles = {
        True: keep(_check_box("#ffffff", "#ffffff", theme.ACCENT)),
        False: keep(_check_box(theme.ENTRY, theme.FG_DIM)),
    }
    style.configure("ToggleOn.TButton", anchor="w")
    style.configure("ToggleOff.TButton", anchor="w")

    # fix: selected rows rendered with a white background, hiding the text
    style.configure("Treeview", background=theme.ENTRY, fieldbackground=theme.ENTRY,
                    foreground=theme.FG, bordercolor=theme.SEL, borderwidth=0, rowheight=24)
    style.map("Treeview", background=[("selected", theme.ACCENT)], foreground=[("selected", "#ffffff")])
    style.configure("Treeview.Heading", background=theme.PANEL, foreground=theme.FG, relief="flat")
    style.map("Treeview.Heading", background=[("active", theme.SEL)])

    root._rounded_images = images  # keep references; Tk drops PhotoImages once they are garbage-collected
