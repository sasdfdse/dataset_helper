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

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        video_frame = ttk.Frame(notebook)
        aug_frame = ttk.Frame(notebook)

        notebook.add(video_frame, text="  📹 Video Tool  ")
        notebook.add(aug_frame, text="  🔧 Augmentation  ")

        VideoTab(video_frame).pack(fill=tk.BOTH, expand=True)
        AugmentTab(aug_frame).pack(fill=tk.BOTH, expand=True)
