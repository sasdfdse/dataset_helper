import signal
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from gui import app_prefs, rounded, theme, vertical_filedialog
from gui.video_tab import VideoTab
from gui.augment_tab import AugmentTab
from gui.background_tab import BackgroundTab


_SIGNAL_POLL_MS = 200


def install_sigint(root: tk.Tk) -> None:
    """터미널의 Ctrl+C로 창을 닫는다.

    Tk 이벤트 루프(C 코드)가 도는 동안에는 파이썬 신호 처리기가 실행되지 않아 Ctrl+C가 먹지
    않는다. 그래서 짧은 주기로 빈 콜백을 돌려 파이썬에 제어를 넘기고, 처리기는 창을 정상
    종료(destroy)한다. destroy는 실행 중인 워커를 멈추고 UI 상태를 저장하는 정리 코드를 거친다.
    """
    def on_sigint(signum, frame):
        signal.signal(signal.SIGINT, signal.default_int_handler)   # a second Ctrl+C aborts immediately
        print("\nCtrl+C: 창을 닫습니다", file=sys.stderr)
        root.after_idle(_destroy_quietly, root)

    def tick():
        try:
            root.after(_SIGNAL_POLL_MS, tick)
        except tk.TclError:
            pass

    signal.signal(signal.SIGINT, on_sigint)
    tick()


def _destroy_quietly(root: tk.Tk) -> None:
    try:
        root.destroy()
    except tk.TclError:
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Helper — ROBIT")
        self.geometry(f"1150x{min(840, self.winfo_screenheight() - 80)}")
        self.minsize(900, 600)
        vertical_filedialog.install(self)
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self._apply_theme(style)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        video_frame = ttk.Frame(notebook)
        aug_frame = ttk.Frame(notebook)
        bg_frame = ttk.Frame(notebook)

        notebook.add(video_frame, text="  📹 Video Tool  ")
        notebook.add(aug_frame, text="  🔧 Augmentation  ")
        notebook.add(bg_frame, text="  🖼 Background Swap  ")

        VideoTab(video_frame).pack(fill=tk.BOTH, expand=True)
        AugmentTab(aug_frame).pack(fill=tk.BOTH, expand=True)
        bg_tab = BackgroundTab(bg_frame)
        bg_tab.pack(fill=tk.BOTH, expand=True)
        bg_tab.start_hardware_probe()

        self._root_btn = ttk.Button(self, command=self._choose_default_root)
        self._root_btn.place(relx=1.0, x=-12, y=8, anchor="ne")
        self._refresh_root_btn()

    def _refresh_root_btn(self) -> None:
        root = app_prefs.default_root()
        self._root_btn.configure(text=f"📁 기본 폴더: {root.name or root}" if root else "📁 기본 폴더 지정")

    def _choose_default_root(self) -> None:
        current = app_prefs.default_root()
        path = filedialog.askdirectory(
            parent=self, title="기본 폴더 선택 — 모든 Browse 창이 이 폴더에서 열립니다",
            initialdir=str(current or Path.home()))
        if path:
            app_prefs.set_default_root(Path(path))
            self._refresh_root_btn()

    def _apply_theme(self, style: ttk.Style):
        BG, PANEL, ACCENT, ACCENT2 = theme.BG, theme.PANEL, theme.ACCENT, theme.ACCENT2
        FG, FG_DIM, ENTRY, SEL = theme.FG, theme.FG_DIM, theme.ENTRY, theme.SEL

        self.configure(bg=BG)

        style.configure(".",
            background=BG,
            foreground=FG,
            fieldbackground=ENTRY,
            troughcolor=PANEL,
            bordercolor=SEL,
            darkcolor=PANEL,
            lightcolor=PANEL,
            font=("Segoe UI", 10),
        )

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("TRadiobutton", background=BG, foreground=FG)
        style.configure("TSeparator", background=SEL)

        style.configure("TLabelframe",
            background=BG,
            foreground=ACCENT,
            bordercolor=SEL,
            relief="flat",
        )
        style.configure("TLabelframe.Label",
            background=BG,
            foreground=ACCENT,
            font=("Segoe UI", 10, "bold"),
        )

        style.configure("TButton",
            background=ACCENT,
            foreground="#ffffff",
            bordercolor=ACCENT,
            focuscolor=ACCENT2,
            relief="flat",
            padding=(10, 5),
        )
        style.map("TButton",
            background=[("active", ACCENT2), ("disabled", SEL)],
            foreground=[("disabled", FG_DIM)],
        )

        style.configure("ToggleOn.TButton",
            background=ACCENT,
            foreground="#ffffff",
            bordercolor=ACCENT,
            relief="flat",
            padding=(10, 5),
        )
        style.map("ToggleOn.TButton",
            background=[("active", ACCENT2)],
        )

        style.configure("ToggleOff.TButton",
            background=ENTRY,
            foreground=FG_DIM,
            bordercolor=SEL,
            relief="flat",
            padding=(10, 5),
        )
        style.map("ToggleOff.TButton",
            background=[("active", SEL)],
        )

        style.configure("TEntry",
            fieldbackground=ENTRY,
            foreground=FG,
            bordercolor=SEL,
            insertcolor=FG,
            relief="flat",
        )

        style.configure("TSpinbox",
            fieldbackground=ENTRY,
            foreground=FG,
            bordercolor=SEL,
            arrowcolor=FG,
        )

        style.configure("TScale",
            background=BG,
            troughcolor=ENTRY,
            sliderrelief="flat",
        )
        style.map("TScale",
            background=[("active", ACCENT)],
        )

        style.configure("Horizontal.TProgressbar",
            troughcolor=ENTRY,
            background=ACCENT,
            bordercolor=SEL,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

        style.configure("TNotebook",
            background=PANEL,
            bordercolor=SEL,
            tabmargins=[2, 4, 0, 0],
        )
        style.configure("TNotebook.Tab",
            background=PANEL,
            foreground=FG_DIM,
            padding=[12, 6],
            font=("Segoe UI", 10),
        )
        style.map("TNotebook.Tab",
            background=[("selected", BG)],
            foreground=[("selected", ACCENT)],
            font=[("selected", ("Segoe UI", 10, "bold"))],
        )

        style.configure("TCombobox",
            fieldbackground=ENTRY,
            foreground=FG,
            selectbackground=SEL,
        )

        self.option_add("*Toplevel.background", BG)
        self.option_add("*TCombobox*Listbox.background", ENTRY)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        rounded.install(self)
