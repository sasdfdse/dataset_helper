"""keep 항목 추가/편집 모달 다이얼로그 (Background Swap 탭 전용, torch 비의존)."""
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from background_swap.stages.color_region import hsv_range_from_click
from background_swap.common.settings import KeepItem
from gui.background_state import to_number as _to_number
from gui import theme

_PANEL_BG = theme.ENTRY


def _hsv_swatch_hex(lower, upper) -> str:
    """범위 중앙 HSV를 #RRGGBB 로 (견본용)."""
    h = int(round((lower[0] + upper[0]) / 2)) % 180
    s = int(np.clip(round((lower[1] + upper[1]) / 2), 0, 255))
    v = int(np.clip(round((lower[2] + upper[2]) / 2), 0, 255))
    b, g, r = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


class KeepItemDialog(tk.Toplevel):
    """keep 항목 추가/편집 모달 다이얼로그. 확정되면 `self.result`에 KeepItem."""

    def __init__(self, tab, item: Optional[KeepItem],
                 existing_names: list[str]):
        super().__init__(tab)
        self.title("유지 항목 편집" if item else "유지 항목 추가")
        self.transient(tab.winfo_toplevel())
        self.resizable(False, False)
        self.result: Optional[KeepItem] = None
        self._tab = tab
        self._existing = existing_names
        d = item or KeepItem(name="", method="color")
        self._lower, self._upper = d.hsv_lower, d.hsv_upper

        self._name_var = tk.StringVar(value=d.name)
        self._method = tk.StringVar(value=d.method)
        self._required = tk.BooleanVar(value=d.required)
        self._prompt = tk.StringVar(value=d.prompt)
        self._tol = {k: tk.StringVar(value=str(getattr(d, k))) for k in ("h_tol", "s_tol", "v_tol")}
        self._largest = tk.BooleanVar(value=d.largest_only)
        self._hull = tk.BooleanVar(value=d.convex_hull)
        self._min_area = tk.StringVar(value=str(d.min_area_ratio))

        body = ttk.Frame(self, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text="이름:").grid(row=0, column=0, sticky="w")
        ttk.Entry(body, textvariable=self._name_var, width=28).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(body, text="방식:").grid(row=1, column=0, sticky="w")
        mf = ttk.Frame(body)
        mf.grid(row=1, column=1, sticky="w")
        for val in ("color", "text"):
            ttk.Radiobutton(mf, text=val, value=val, variable=self._method,
                            command=self._sync_method).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(body, text="필수 (못 찾으면 이미지 스킵)", variable=self._required).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2)

        self._color_frame = ttk.Frame(body)
        self._color_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        cf = self._color_frame
        cf.columnconfigure(1, weight=1)
        pick = ttk.Frame(cf)
        pick.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Button(pick, text="미리보기에서 클릭", command=self._start_pick).pack(side=tk.LEFT)
        self._swatch = tk.Label(pick, width=6, relief="solid", borderwidth=1)
        self._swatch.pack(side=tk.LEFT, padx=8)
        self._swatch_txt = ttk.Label(pick, text="")
        self._swatch_txt.pack(side=tk.LEFT)
        for r, (key, label) in enumerate((("h_tol", "H tol"), ("s_tol", "S tol"), ("v_tol", "V tol")), 1):
            ttk.Label(cf, text=f"{label}:").grid(row=r, column=0, sticky="w")
            ttk.Spinbox(cf, from_=0, to=255, textvariable=self._tol[key], width=6).grid(
                row=r, column=1, sticky="w", pady=1)
        ttk.Checkbutton(cf, text="largest_only (가장 큰 영역만)", variable=self._largest).grid(
            row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(cf, text="convex_hull (볼록 껍질)", variable=self._hull).grid(
            row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(cf, text="min_area_ratio:").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(cf, from_=0.0, to=1.0, increment=0.01, textvariable=self._min_area,
                    width=6).grid(row=6, column=1, sticky="w", pady=1)

        self._text_frame = ttk.Frame(body)
        self._text_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        self._text_frame.columnconfigure(1, weight=1)
        ttk.Label(self._text_frame, text="프롬프트:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self._text_frame, textvariable=self._prompt, width=28).grid(
            row=0, column=1, sticky="ew", pady=2)

        btns = ttk.Frame(body)
        btns.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="OK", command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.LEFT)

        self._sync_method()
        self._update_swatch()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
        self._grab(20)

    def _grab(self, tries: int) -> None:
        """창이 보이게 된 뒤 모달 grab (보이기 전에는 TclError → 재시도)."""
        if not self.winfo_exists():
            return
        try:
            self.grab_set()
        except tk.TclError:
            if tries > 0:
                self.after(50, self._grab, tries - 1)

    def _sync_method(self) -> None:
        if self._method.get() == "color":
            self._text_frame.grid_remove()
            self._color_frame.grid()
        else:
            self._color_frame.grid_remove()
            self._text_frame.grid()

    def _update_swatch(self) -> None:
        if self._lower and self._upper:
            self._swatch.configure(bg=_hsv_swatch_hex(self._lower, self._upper))
            self._swatch_txt.configure(text=f"HSV {self._lower} ~ {self._upper}")
        else:
            self._swatch.configure(bg=_PANEL_BG)
            self._swatch_txt.configure(text="(미선택)")

    def _tol_values(self) -> Optional[tuple[int, int, int]]:
        vals = [_to_number(self._tol[k].get(), int) for k in ("h_tol", "s_tol", "v_tol")]
        if any(v is None or v < 0 for v in vals):
            return None
        return tuple(vals)

    def _start_pick(self) -> None:
        tols = self._tol_values()
        if tols is None:
            messagebox.showerror("Error", "H/S/V tol 은 0 이상의 숫자여야 합니다.", parent=self)
            return
        if not self._tab.has_preview_image():
            messagebox.showerror("Error", "먼저 Images/Labels 폴더를 지정해 미리보기를 표시하세요.",
                                 parent=self)
            return
        # The modal grab blocks clicks on the main window, so release it and hide while picking.
        self.grab_release()
        self.withdraw()
        self._tab.begin_pick(lambda img, x, y: self._on_picked(img, x, y, tols), self._pick_done)

    def _on_picked(self, img: np.ndarray, x: int, y: int, tols: tuple[int, int, int]) -> None:
        self._lower, self._upper = hsv_range_from_click(img, x, y, *tols)
        self._update_swatch()

    def _pick_done(self) -> None:
        if self.winfo_exists():
            self.deiconify()
            self._grab(20)

    def _ok(self) -> None:
        name = self._name_var.get().strip()
        method = self._method.get()
        if not name:
            messagebox.showerror("Error", "이름을 입력하세요.", parent=self)
            return
        if name in self._existing:
            messagebox.showerror("Error", f"같은 이름의 항목이 이미 있습니다: {name}", parent=self)
            return
        item = KeepItem(name=name, method=method, required=self._required.get())
        if method == "color":
            if not (self._lower and self._upper):
                messagebox.showerror("Error", "미리보기를 클릭해 색을 선택하세요.", parent=self)
                return
            tols = self._tol_values()
            area = _to_number(self._min_area.get(), float)
            if tols is None or area is None or not 0.0 <= area <= 1.0:
                messagebox.showerror("Error", "tol(0 이상 숫자)과 min_area_ratio(0~1)를 확인하세요.",
                                     parent=self)
                return
            item.hsv_lower, item.hsv_upper = list(self._lower), list(self._upper)
            item.h_tol, item.s_tol, item.v_tol = tols
            item.largest_only, item.convex_hull = self._largest.get(), self._hull.get()
            item.min_area_ratio = area
        else:
            if not self._prompt.get().strip():
                messagebox.showerror("Error", "프롬프트를 입력하세요.", parent=self)
                return
            item.prompt = self._prompt.get().strip()
        self.result = item
        self.destroy()
