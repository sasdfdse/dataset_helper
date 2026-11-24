"""Background Swap 탭 보조 창: 3지선다 모달, 리포트 창."""
from typing import Optional

import tkinter as tk
from tkinter import ttk
from gui import theme


class ChoiceDialog(tk.Toplevel):
    """메시지 + 버튼 N개 모달. 선택된 키가 `self.result`에 (창 닫기/Esc는 마지막 선택지의 키)."""

    def __init__(self, parent, title: str, message: str, choices: list[tuple[str, str]]):
        super().__init__(parent)
        self.title(title)
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self._default = choices[-1][0]
        self.result: str = self._default
        ttk.Label(self, text=message, wraplength=420, justify="left").pack(padx=14, pady=(14, 8))
        row = ttk.Frame(self)
        row.pack(pady=(0, 12))
        for key, text in choices:
            ttk.Button(row, text=text, command=lambda k=key: self._choose(k)).pack(side=tk.LEFT, padx=4)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
        self._grab(20)

    def _choose(self, key: str) -> None:
        self.result = key
        self.destroy()

    def _grab(self, tries: int) -> None:
        if not self.winfo_exists():
            return
        try:
            self.grab_set()
        except tk.TclError:
            if tries > 0:
                self.after(50, self._grab, tries - 1)


def ask_choice(parent, title: str, message: str, choices: list[tuple[str, str]]) -> str:
    dlg = ChoiceDialog(parent, title, message, choices)
    parent.wait_window(dlg)
    return dlg.result


class ReportWindow(tk.Toplevel):
    def __init__(self, parent, text: str, title: str = "Background Swap Report",
                 size: str = "520x520"):
        super().__init__(parent)
        self.title(title)
        self.geometry(size)
        box = tk.Text(self, wrap="word", bg=theme.ENTRY, fg=theme.FG, relief="flat", padx=8, pady=8)
        sb = ttk.Scrollbar(self, orient="vertical", command=box.yview)
        box.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        box.insert("1.0", text)
        box.tag_configure("heading", foreground=theme.ACCENT)
        for i, line in enumerate(text.splitlines(), 1):
            if line.startswith("["):
                box.tag_add("heading", f"{i}.0", f"{i}.end")
        box.configure(state="disabled")
        self.text_widget: Optional[tk.Text] = box
