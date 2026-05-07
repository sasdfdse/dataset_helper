import tkinter as tk
from tkinter import ttk

from gui.video_tab import VideoTab
from gui.augment_tab import AugmentTab


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Helper — ROBIT")
        self.geometry("1150x720")
        self.minsize(900, 600)
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self._apply_theme(style)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        video_frame = ttk.Frame(notebook)
        aug_frame = ttk.Frame(notebook)

        notebook.add(video_frame, text="  📹 Video Tool  ")
        notebook.add(aug_frame, text="  🔧 Augmentation  ")

        VideoTab(video_frame).pack(fill=tk.BOTH, expand=True)
        AugmentTab(aug_frame).pack(fill=tk.BOTH, expand=True)

    def _apply_theme(self, style: ttk.Style):
        BG      = "#1e1e2e"   # 배경 (어두운 남색)
        PANEL   = "#2a2a3e"   # 패널/프레임
        ACCENT  = "#7c5cfc"   # 강조색 (바이올렛)
        ACCENT2 = "#5a3fcf"   # 버튼 hover
        FG      = "#cdd6f4"   # 텍스트
        FG_DIM  = "#6c7086"   # 비활성 텍스트
        ENTRY   = "#313244"   # 입력 필드
        SEL     = "#45475a"   # 선택/hover

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
