import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from core.label import BBox, read_labels
from core.pipeline import AugConfig, run_batch, preview_augment, _collect_originals


_PREVIEW_SIZE = (380, 280)
_BBOX_COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44",
    "#FF44FF", "#44FFFF", "#FF8844", "#44FF88",
]


class _AugToggle:
    """Toggle button that clearly shows ON / OFF state."""

    def __init__(self, parent: ttk.Frame, label: str, var: tk.BooleanVar,
                 on_change_cb):
        self._label = label
        self._var = var
        self._on_change_cb = on_change_cb
        self._btn = ttk.Button(parent, command=self._click, width=16)
        self._refresh()

    def _click(self) -> None:
        self._var.set(not self._var.get())
        self._refresh()
        self._on_change_cb()

    def _refresh(self) -> None:
        if self._var.get():
            self._btn.configure(text=f"✔  {self._label}", style="ToggleOn.TButton")
        else:
            self._btn.configure(text=f"     {self._label}", style="ToggleOff.TButton")

    def grid(self, **kwargs) -> None:
        self._btn.grid(**kwargs)


class AugmentTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._originals: List[Path] = []
        self._current_idx: int = 0
        self._result_pairs: List[Tuple[Path, Path, Path, Path]] = []
        self._results_mode: bool = False
        self._running: bool = False
        self._build_ui()

    # ── Layout ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0, minsize=250)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_settings(self)
        self._build_preview(self)

    def _build_settings(self, parent) -> None:
        outer = ttk.LabelFrame(parent, text="Settings", padding=4)
        outer.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0, width=260)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        frame = ttk.Frame(canvas, padding=4)
        frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(frame_id, width=event.width)

        frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(frame, text="images/ folder:").grid(row=0, column=0, sticky="w")
        self._images_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._images_var, width=22).grid(row=1, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_images).grid(row=1, column=1, padx=(4, 0))

        ttk.Label(frame, text="labels/ folder:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._labels_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._labels_var, width=22).grid(row=3, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_labels).grid(row=3, column=1, padx=(4, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

        row = 5
        row = self._aug_section(frame, row, "Brightness", "brightness",
                                [("min", -100, 100, -50), ("max", -100, 100, 50)])
        row = self._aug_section(frame, row, "Blur", "blur",
                                [("radius", 1, 15, 3)])
        row = self._aug_section(frame, row, "Noise", "noise",
                                [("strength", 0, 100, 25)])
        row = self._aug_section(frame, row, "Scale", "scale",
                                [("min ×", 50, 200, 80), ("max ×", 50, 200, 120)])
        row = self._aug_section(frame, row, "Flip H", "flip_h", [])
        row = self._aug_section(frame, row, "Flip V", "flip_v", [])
        row = self._aug_section(frame, row, "Rotation", "rotation",
                                [("min_deg", -180, 180, -45), ("max_deg", -180, 180, 45)])
        row = self._aug_section(frame, row, "Crop", "crop",
                                [("min_pct", 50, 99, 70), ("max_pct", 50, 99, 95)])
        row = self._aug_section(frame, row, "Shear", "shear",
                                [("min_deg", -45, 45, -15), ("max_deg", -45, 45, 15)])
        row = self._aug_section(frame, row, "Grayscale", "grayscale", [])
        row = self._aug_section(frame, row, "Saturation", "saturation",
                                [("min_pct", 0, 200, 50), ("max_pct", 0, 200, 150)])
        row = self._aug_section(frame, row, "Exposure", "exposure",
                                [("min_pct", 0, 200, 50), ("max_pct", 0, 200, 150)])
        row = self._aug_section(frame, row, "Motion Blur", "motion_blur",
                                [("kernel", 3, 31, 15), ("ang_min", 0, 360, 0), ("ang_max", 0, 360, 360)])
        row = self._aug_section(frame, row, "Camera Gain", "camera_gain",
                                [("min_pct", 0, 200, 80), ("max_pct", 0, 200, 120)])
        row = self._aug_section(frame, row, "CutMix", "cutmix",
                                [("pairs", 1, 100, 10)])
        row = self._aug_section(frame, row, "Mosaic", "mosaic",
                                [("pairs", 1, 100, 10)])

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        ttk.Label(frame, text="Multiplier (×):").grid(row=row, column=0, sticky="w")
        self._multiplier_var = tk.IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=50, textvariable=self._multiplier_var, width=6).grid(
            row=row, column=1, sticky="w")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(2, weight=0)

    def _aug_section(self, parent, row: int, label: str, key: str, params) -> int:
        enabled_var = tk.BooleanVar(value=False)
        setattr(self, f"_{key}_enabled", enabled_var)

        toggle = _AugToggle(parent, label, enabled_var, self._on_control_changed)
        toggle.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        row += 1

        for name, lo, hi, default in params:
            ttk.Label(parent, text=f"  {name}:").grid(row=row, column=0, sticky="w")
            var = tk.IntVar(value=default)
            attr = f"_{key}_{name.replace(' ', '_').replace('×', 'x')}_var"
            setattr(self, attr, var)
            var.trace_add("write", self._on_control_changed)
            ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal").grid(
                row=row, column=1, sticky="ew")
            ttk.Label(parent, textvariable=var, width=4, anchor="e").grid(
                row=row, column=2, sticky="e", padx=(2, 0))
            row += 1
        return row

    def _build_preview(self, parent) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)

        img_frame = ttk.LabelFrame(frame, text="Preview", padding=4)
        img_frame.grid(row=0, column=0, sticky="nsew")
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)

        self._orig_title_lbl = ttk.Label(img_frame, text="Original")
        self._orig_title_lbl.grid(row=0, column=0)
        self._aug_title_lbl = ttk.Label(img_frame, text="Augmented (실시간 미리보기)")
        self._aug_title_lbl.grid(row=0, column=1)

        self._orig_lbl = ttk.Label(img_frame, background="#313244")
        self._orig_lbl.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")

        self._aug_lbl = ttk.Label(img_frame, background="#313244")
        self._aug_lbl.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")

        nav_frame = ttk.Frame(img_frame)
        nav_frame.grid(row=2, column=0, columnspan=2)
        ttk.Button(nav_frame, text="◀ Prev", command=self._prev_image).pack(side=tk.LEFT, padx=8)
        self._nav_lbl = ttk.Label(nav_frame, text="–")
        self._nav_lbl.pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next ▶", command=self._next_image).pack(side=tk.LEFT, padx=8)

        bottom = ttk.Frame(frame)
        bottom.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(bottom, variable=self._progress_var, maximum=100).grid(
            row=0, column=0, sticky="ew", padx=4)

        self._status_lbl = ttk.Label(bottom, text="images/ 폴더를 선택하세요")
        self._status_lbl.grid(row=1, column=0)

        self._run_btn = ttk.Button(bottom, text="▶ Run Augmentation", command=self._on_run)
        self._run_btn.grid(row=2, column=0, pady=6)

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_images(self) -> None:
        path = filedialog.askdirectory(title="Select images/ folder")
        if path:
            self._images_var.set(path)
            self._images_dir = Path(path)
            self._results_mode = False
            self._load_originals()

    def _browse_labels(self) -> None:
        path = filedialog.askdirectory(title="Select labels/ folder")
        if path:
            self._labels_var.set(path)
            self._labels_dir = Path(path)
            self._refresh_display()

    # ── Originals loading ────────────────────────────────────────────

    def _load_originals(self) -> None:
        if self._images_dir is None:
            return
        self._originals = _collect_originals(self._images_dir)
        self._current_idx = 0
        if self._originals:
            self._status_lbl.configure(text=f"원본 {len(self._originals)}장 — 슬라이더를 조정해 미리보기")
            self._refresh_display()
        else:
            self._status_lbl.configure(text="이미지 없음")

    # ── Live preview ─────────────────────────────────────────────────

    def _on_control_changed(self, *args) -> None:
        if not self._results_mode:
            self._refresh_display()

    def _refresh_display(self) -> None:
        if not self._originals:
            return

        img_path = self._originals[self._current_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return

        lbl_path = self._labels_dir / (img_path.stem + ".txt") if self._labels_dir else None
        boxes = read_labels(lbl_path) if lbl_path and lbl_path.exists() else []

        # Left: original
        orig_photo = self._to_photoimage(self._draw_bboxes(img, boxes))
        self._orig_lbl.configure(image=orig_photo)
        self._orig_lbl.image = orig_photo

        # Right: live augmented preview
        config = self._build_config()
        aug_img, aug_boxes = preview_augment(img.copy(), list(boxes), config)
        aug_photo = self._to_photoimage(self._draw_bboxes(aug_img, aug_boxes))
        self._aug_lbl.configure(image=aug_photo)
        self._aug_lbl.image = aug_photo

        total = len(self._originals)
        self._nav_lbl.configure(text=f"{self._current_idx + 1} / {total}")

    # ── Navigation ───────────────────────────────────────────────────

    def _prev_image(self) -> None:
        if self._results_mode:
            self._prev_result()
            return
        if not self._originals:
            return
        self._current_idx = (self._current_idx - 1) % len(self._originals)
        self._refresh_display()

    def _next_image(self) -> None:
        if self._results_mode:
            self._next_result()
            return
        if not self._originals:
            return
        self._current_idx = (self._current_idx + 1) % len(self._originals)
        self._refresh_display()

    def _prev_result(self) -> None:
        if not self._result_pairs:
            return
        self._current_idx = (self._current_idx - 1) % len(self._result_pairs)
        self._show_result(self._current_idx)

    def _next_result(self) -> None:
        if not self._result_pairs:
            return
        self._current_idx = (self._current_idx + 1) % len(self._result_pairs)
        self._show_result(self._current_idx)

    def _show_result(self, idx: int) -> None:
        orig_path, orig_lbl, aug_path, aug_lbl = self._result_pairs[idx]
        orig_img = cv2.imread(str(orig_path))
        aug_img = cv2.imread(str(aug_path))
        orig_boxes = read_labels(orig_lbl)
        aug_boxes = read_labels(aug_lbl)

        orig_photo = self._to_photoimage(self._draw_bboxes(orig_img, orig_boxes))
        aug_photo = self._to_photoimage(self._draw_bboxes(aug_img, aug_boxes))
        self._orig_lbl.configure(image=orig_photo)
        self._orig_lbl.image = orig_photo
        self._aug_lbl.configure(image=aug_photo)
        self._aug_lbl.image = aug_photo

        total = len(self._result_pairs)
        self._nav_lbl.configure(text=f"{idx + 1} / {total}")

    # ── Run ──────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        if self._running:
            return
        if self._images_dir is None or self._labels_dir is None:
            messagebox.showerror("Error", "images/ 와 labels/ 폴더를 모두 선택하세요.")
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
            flip_h_enabled=self._flip_h_enabled.get(),
            flip_v_enabled=self._flip_v_enabled.get(),
            rotation_enabled=self._rotation_enabled.get(),
            rotation_min=self._rotation_min_deg_var.get(),
            rotation_max=self._rotation_max_deg_var.get(),
            crop_enabled=self._crop_enabled.get(),
            crop_min=self._crop_min_pct_var.get(),
            crop_max=self._crop_max_pct_var.get(),
            shear_enabled=self._shear_enabled.get(),
            shear_min=self._shear_min_deg_var.get(),
            shear_max=self._shear_max_deg_var.get(),
            grayscale_enabled=self._grayscale_enabled.get(),
            saturation_enabled=self._saturation_enabled.get(),
            saturation_min=self._saturation_min_pct_var.get(),
            saturation_max=self._saturation_max_pct_var.get(),
            exposure_enabled=self._exposure_enabled.get(),
            exposure_min=self._exposure_min_pct_var.get(),
            exposure_max=self._exposure_max_pct_var.get(),
            motion_blur_enabled=self._motion_blur_enabled.get(),
            motion_blur_kernel=self._motion_blur_kernel_var.get(),
            motion_blur_angle_min=self._motion_blur_ang_min_var.get(),
            motion_blur_angle_max=self._motion_blur_ang_max_var.get(),
            camera_gain_enabled=self._camera_gain_enabled.get(),
            camera_gain_min=self._camera_gain_min_pct_var.get(),
            camera_gain_max=self._camera_gain_max_pct_var.get(),
            mosaic_enabled=self._mosaic_enabled.get(),
            mosaic_pairs=self._mosaic_pairs_var.get(),
        )

    def _run_worker(self, config: AugConfig) -> None:
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

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = done / total * 100
        else:
            pct = 100.0
        self.after(0, self._update_progress, pct, done, total)

    def _update_progress(self, pct: float, done: int, total: int) -> None:
        self._progress_var.set(pct)
        self._status_lbl.configure(text=f"{done} / {total}")

    def _on_complete(self) -> None:
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="완료! ← Prev / Next ▶ 로 결과 확인")
        self._aug_title_lbl.configure(text="Augmented (결과)")
        self._load_result_pairs()

    def _on_error(self, msg: str) -> None:
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="오류 발생")
        messagebox.showerror("Augmentation Error", msg)

    # ── Result pairs ─────────────────────────────────────────────────

    def _load_result_pairs(self) -> None:
        if self._images_dir is None:
            return
        aug_imgs = sorted(
            p for p in self._images_dir.glob("*.jpg")
            if "_aug" in p.stem or "_cutmix" in p.stem or "_mosaic" in p.stem
        )
        self._result_pairs = []
        for aug_path in aug_imgs:
            aug_lbl = self._labels_dir / (aug_path.stem + ".txt")
            if "_cutmix" in aug_path.stem or "_mosaic" in aug_path.stem:
                orig_path = aug_path
                orig_lbl = aug_lbl
            else:
                orig_stem = aug_path.stem.rsplit("_aug", 1)[0]
                orig_path = self._images_dir / (orig_stem + aug_path.suffix)
                orig_lbl = self._labels_dir / (orig_stem + ".txt")
            self._result_pairs.append((orig_path, orig_lbl, aug_path, aug_lbl))

        self._results_mode = True
        self._current_idx = 0
        if self._result_pairs:
            self._show_result(0)

    # ── Image helpers ─────────────────────────────────────────────────

    def _draw_bboxes(self, img: np.ndarray, boxes: List[BBox]) -> np.ndarray:
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        result = img.copy()
        h, w = result.shape[:2]
        for i, b in enumerate(boxes):
            color_hex = _BBOX_COLORS[i % len(_BBOX_COLORS)]
            color_bgr = tuple(int(color_hex[j:j + 2], 16) for j in (5, 3, 1))
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
