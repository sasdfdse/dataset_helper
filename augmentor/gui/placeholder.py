"""입력 칸의 회색 안내 글자 (placeholder).

입력 칸 위에 흐린 라벨을 겹쳐 보여 줄 뿐 칸의 값(textvariable)에는 들어가지 않는다.
칸을 클릭(포커스)하면 사라지고, 칸이 빈 채로 포커스가 빠지면 다시 나타난다.
"""
import tkinter as tk
from tkinter import ttk

from gui import theme

_PAD_X = 6      # aligns with the text start in the field (RField border + inner padding)


class Placeholder:
    def __init__(self, entry: ttk.Entry, var: tk.StringVar, text: str) -> None:
        self.entry, self.var, self.text = entry, var, text
        self.label = tk.Label(entry, text=text, anchor="w", bg=theme.ENTRY, fg=theme.FG_DIM,
                              font=entry.cget("font") or "TkTextFont", borderwidth=0,
                              padx=0, pady=0, cursor="xterm")
        self._focused = False
        self.label.bind("<Button-1>", self._click)
        entry.bind("<FocusIn>", lambda e: self._set_focus(True), add="+")
        entry.bind("<FocusOut>", lambda e: self._set_focus(False), add="+")
        self._trace = var.trace_add("write", lambda *a: self.update())
        entry.bind("<Destroy>", self._on_destroy, add="+")
        self.update()

    @property
    def visible(self) -> bool:
        return bool(self.label.winfo_manager())

    def _click(self, event=None):
        if not self.entry.instate(["disabled"]):
            self.entry.focus_set()
        return "break"

    def set_text(self, text: str) -> None:
        self.text = text
        if self.label.winfo_exists():
            self.label.configure(text=text)
        self.update()

    def _set_focus(self, focused: bool) -> None:
        self._focused = focused
        self.update()

    def update(self) -> None:
        if not self.label.winfo_exists():
            return
        if self.text and not self.var.get() and not self._focused:
            self.label.configure(fg=theme.SEL if self.entry.instate(["disabled"]) else theme.FG_DIM)
            self.label.place(x=_PAD_X, rely=0.5, anchor="w", relwidth=1.0, width=-2 * _PAD_X)
        else:
            self.label.place_forget()

    def _on_destroy(self, event) -> None:
        if event.widget is self.entry:
            try:
                self.var.trace_remove("write", self._trace)
            except tk.TclError:
                pass
