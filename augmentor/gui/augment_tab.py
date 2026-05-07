import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

from core.label import BBox, read_labels
from core.pipeline import AugConfig, run_batch, _collect_originals


_PREVIEW_SIZE = (380, 280)
_BBOX_COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44",
    "#FF44FF", "#44FFFF", "#FF8844", "#44FF88",
]


class AugmentTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._preview_pairs: List[Tuple[Path, Path, Path, Path]] = []
        self._preview_idx: int = 0
        self._running: bool = False
        self._build_ui()

    # ── Layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=240)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_settings(self)
        self._build_preview(self)

    def _build_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="Settings", padding=8)
        frame.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)

        # Folder selection
        ttk.Label(frame, text="images/ folder:").grid(row=0, column=0, sticky="w")
        self._images_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._images_var, width=22).grid(row=1, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_images).grid(row=1, column=1, padx=(4, 0))

        ttk.Label(frame, text="labels/ folder:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._labels_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._labels_var, width=22).grid(row=3, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_labels).grid(row=3, column=1, padx=(4, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)

        # Augmentation controls
        row = 5
        row = self._aug_section(frame, row, "Brightness", "brightness",
                                [("min", -100, 100, -50), ("max", -100, 100, 50)])
        row = self._aug_section(frame, row, "Blur", "blur",
                                [("radius", 1, 15, 3)])
        row = self._aug_section(frame, row, "Noise", "noise",
                                [("strength", 0, 100, 25)])
        row = self._aug_section(frame, row, "Scale", "scale",
                                [("min ×", 50, 200, 80), ("max ×", 50, 200, 120)])
        row = self._aug_section(frame, row, "CutMix", "cutmix",
                                [("pairs", 1, 100, 10)])

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(frame, text="Multiplier (×):").grid(row=row, column=0, sticky="w")
        self._multiplier_var = tk.IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=50, textvariable=self._multiplier_var, width=6).grid(
            row=row, column=1, sticky="w")

        frame.columnconfigure(0, weight=1)

    def _aug_section(self, parent, row: int, label: str, key: str, params) -> int:
        enabled_var = tk.BooleanVar(value=False)
        setattr(self, f"_{key}_enabled", enabled_var)
        ttk.Checkbutton(parent, text=label, variable=enabled_var).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for name, lo, hi, default in params:
            ttk.Label(parent, text=f"  {name}:").grid(row=row, column=0, sticky="w")
            var = tk.IntVar(value=default)
            setattr(self, f"_{key}_{name.replace(' ', '_').replace('×', 'x')}_var", var)
            ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal").grid(
                row=row, column=1, sticky="ew")
            row += 1
        return row

    def _build_preview(self, parent):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)

        # Image preview area
        img_frame = ttk.LabelFrame(frame, text="Preview", padding=4)
        img_frame.grid(row=0, column=0, sticky="nsew")
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)

        ttk.Label(img_frame, text="Original").grid(row=0, column=0)
        ttk.Label(img_frame, text="Augmented").grid(row=0, column=1)

        self._orig_lbl = ttk.Label(img_frame, background="#313244")
        self._orig_lbl.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")

        self._aug_lbl = ttk.Label(img_frame, background="#313244")
        self._aug_lbl.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")

        nav_frame = ttk.Frame(img_frame)
        nav_frame.grid(row=2, column=0, columnspan=2)
        ttk.Button(nav_frame, text="◀ Prev", command=self._prev_preview).pack(side=tk.LEFT, padx=8)
        self._preview_lbl = ttk.Label(nav_frame, text="–")
        self._preview_lbl.pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next ▶", command=self._next_preview).pack(side=tk.LEFT, padx=8)

        # Progress + run
        bottom = ttk.Frame(frame)
        bottom.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(bottom, variable=self._progress_var, maximum=100)
        self._progress_bar.grid(row=0, column=0, sticky="ew", padx=4)

        self._status_lbl = ttk.Label(bottom, text="Ready")
        self._status_lbl.grid(row=1, column=0)

        self._run_btn = ttk.Button(bottom, text="▶ Run Augmentation", command=self._on_run)
        self._run_btn.grid(row=2, column=0, pady=6)

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_images(self):
        path = filedialog.askdirectory(title="Select images/ folder")
        if path:
            self._images_var.set(path)
            self._images_dir = Path(path)

    def _browse_labels(self):
        path = filedialog.askdirectory(title="Select labels/ folder")
        if path:
            self._labels_var.set(path)
            self._labels_dir = Path(path)

    # ── Run ─────────────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        if self._images_dir is None or self._labels_dir is None:
            messagebox.showerror("Error", "Select both images/ and labels/ folders first.")
            return

        config = self._build_config()
        self._running = True
        self._run_btn.configure(state=tk.DISABLED)
        self._progress_var.set(0)
        self._status_lbl.configure(text="Running…")

        thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        thread.start()

    def _build_config(self) -> AugConfig:
        return AugConfig(
            brightness_enabled=self._brightness_enabled.get(),
            brightness_min=self._brightness_min_var.get(),
            brightness_max=self._brightness_max_var.get(),
            blur_enabled=self._blur_enabled.get(),
            blur_radius=self._blur_radius_var.get(),
            noise_enabled=self._noise_enabled.get(),
            noise_strength=float(self._noise_strength_var.get()),
            scale_enabled=self._scale_enabled.get(),
            scale_min=self._scale_min_x_var.get() / 100.0,
            scale_max=self._scale_max_x_var.get() / 100.0,
            cutmix_enabled=self._cutmix_enabled.get(),
            cutmix_pairs=self._cutmix_pairs_var.get(),
            multiplier=self._multiplier_var.get(),
        )

    def _run_worker(self, config: AugConfig):
        try:
            run_batch(
                self._images_dir,
                self._labels_dir,
                config,
                progress_cb=self._on_progress,
            )
            self.after(0, self._on_complete)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _on_progress(self, done: int, total: int):
        if total > 0:
            pct = done / total * 100
        else:
            pct = 100.0
        self.after(0, self._update_progress, pct, done, total)

    def _update_progress(self, pct: float, done: int, total: int):
        self._progress_var.set(pct)
        self._status_lbl.configure(text=f"{done} / {total}")

    def _on_complete(self):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="Done!")
        self._load_preview_list()

    def _on_error(self, msg: str):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="Error")
        messagebox.showerror("Augmentation Error", msg)

    # ── Preview ──────────────────────────────────────────────────────

    def _load_preview_list(self):
        if self._images_dir is None:
            return
        aug_imgs = sorted(
            p for p in self._images_dir.glob("*.jpg")
            if "_aug" in p.stem or "_cutmix" in p.stem
        )
        self._preview_pairs = []
        for aug_path in aug_imgs:
            aug_label = self._labels_dir / (aug_path.stem + ".txt")
            # Find original stem
            if "_cutmix" in aug_path.stem:
                orig_path = aug_path
                orig_label = aug_label
            else:
                orig_stem = aug_path.stem.rsplit("_aug", 1)[0]
                orig_path = self._images_dir / (orig_stem + aug_path.suffix)
                orig_label = self._labels_dir / (orig_stem + ".txt")
            self._preview_pairs.append((orig_path, orig_label, aug_path, aug_label))

        self._preview_idx = 0
        if self._preview_pairs:
            self._show_preview(0)

    def _show_preview(self, idx: int):
        orig_img_path, orig_lbl_path, aug_img_path, aug_lbl_path = self._preview_pairs[idx]
        orig_img = cv2.imread(str(orig_img_path))
        aug_img = cv2.imread(str(aug_img_path))
        orig_boxes = read_labels(orig_lbl_path)
        aug_boxes = read_labels(aug_lbl_path)

        orig_photo = self._to_photoimage(self._draw_bboxes(orig_img, orig_boxes))
        aug_photo = self._to_photoimage(self._draw_bboxes(aug_img, aug_boxes))

        self._orig_lbl.configure(image=orig_photo)
        self._orig_lbl.image = orig_photo
        self._aug_lbl.configure(image=aug_photo)
        self._aug_lbl.image = aug_photo

        total = len(self._preview_pairs)
        self._preview_lbl.configure(text=f"{idx + 1} / {total}")

    def _draw_bboxes(self, img: np.ndarray, boxes: List[BBox]) -> np.ndarray:
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        result = img.copy()
        h, w = result.shape[:2]
        for i, b in enumerate(boxes):
            color_hex = _BBOX_COLORS[i % len(_BBOX_COLORS)]
            color_bgr = tuple(int(color_hex[j:j+2], 16) for j in (5, 3, 1))
            x1 = int((b.x - b.w / 2) * w)
            y1 = int((b.y - b.h / 2) * h)
            x2 = int((b.x + b.w / 2) * w)
            y2 = int((b.y + b.h / 2) * h)
            cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 2)
        return result

    def _to_photoimage(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize(_PREVIEW_SIZE, Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _prev_preview(self):
        if not self._preview_pairs:
            return
        self._preview_idx = (self._preview_idx - 1) % len(self._preview_pairs)
        self._show_preview(self._preview_idx)

    def _next_preview(self):
        if not self._preview_pairs:
            return
        self._preview_idx = (self._preview_idx + 1) % len(self._preview_pairs)
        self._show_preview(self._preview_idx)
