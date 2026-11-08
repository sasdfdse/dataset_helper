"""Background Swap 탭: 설정 패널, keep 항목 편집, 즉시 마스크 미리보기,
Run/Preview/Stop/Export(서브프로세스 워커), 진행 표시, 결과 4패널 탐색, Report.

torch/모델 코드는 import하지 않는다. 워커는 gui.worker_process, 결과/문구 로직은 gui.background_results.
"""
import copy
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

from background_swap.classes import find_class_names_file
from background_swap.composite import draw_boxes
from background_swap.inputs import collect_items, load_image_bgr
from background_swap.settings import BackgroundSwapSettings, KeepItem, ModelIds
from background_swap.store import reset_output_dir
from core.label import read_labels
from gui import background_state
from gui.augment_tab import _PREVIEW_SIZE
from gui.background_state import (format_axis_values, load_ui_state,
                                  parse_axis_values, quick_mask_preview,
                                  save_ui_state, to_number as _to_number)
from gui.background_results import (ResultList, build_result_list, describe_record,
                                    index_originals, load_panels)
from gui.background_run import PREVIEW_N, RunMixin
from gui.keep_item_dialog import KeepItemDialog
from gui.worker_process import WorkerProcess

_PLACEHOLDER = "Preview 실행 후 표시"
_DEBOUNCE_MS = 150
_PANEL_BG = "#313244"
_MODEL_LABELS = (("segmenter", "Segmenter"), ("detector", "Detector"), ("llm", "LLM"),
                 ("inpainter", "Inpainter"), ("vlm", "VLM"))

# (settings 필드, 형변환, 라벨, spinbox 범위(from, to, increment) 또는 None)
_DATA_NUM = (("frame_stride", int, "Frame stride", (1, 100, 1)),)
_GEN_NUM = (
    ("num_variants", int, "Variants (1-20)", (1, 20, 1)),
    ("max_attempts", int, "Max attempts (1-10)", (1, 10, 1)),
    ("seed", int, "Seed", (0, 2 ** 31 - 1, 1)),
)
_ADV_NUM = (
    ("max_resolution_px", int, "Max resolution px", (256, 4096, 64)),
    ("manual_resolution_px", int, "Manual resolution px", (256, 4096, 64)),
    ("small_box_px", int, "Small box px", (0, 512, 1)),
    ("dilate_px", int, "Dilate px", (0, 100, 1)),
    ("shadow_extend_ratio", float, "Shadow extend ratio", (0.0, 1.0, 0.05)),
    ("feather_px", int, "Feather px", (0, 100, 1)),
    ("min_inpaint_ratio", float, "Min inpaint ratio", (0.0, 1.0, 0.05)),
    ("min_change", float, "Min change", (0.0, 255.0, 1.0)),
)
_NUM_CASTS = {attr: cast for attr, cast, _, _ in _DATA_NUM + _GEN_NUM + _ADV_NUM}
_PREVIEW_ATTRS = ("small_box_px", "dilate_px", "shadow_extend_ratio")
_PATH_ATTRS = ("images_dir", "labels_dir", "output_dir", "class_names_file")


class BackgroundTab(RunMixin, ttk.Frame):
    def __init__(self, parent, state_path: Optional[Path] = None):
        super().__init__(parent)
        self._state_path = state_path
        self._last = BackgroundSwapSettings()      # 마지막으로 유효했던 값 (잘못된 입력 폴백)
        self._keep_items: list[KeepItem] = []
        self._items = []                           # list[ImageItem]
        self._items_key = None
        self._idx = 0
        self._cur_img: Optional[np.ndarray] = None
        self._after_id: Optional[str] = None
        self._pick: Optional[tuple[Callable, Callable]] = None
        self._axis_rows: list[tuple[str, tk.StringVar, ttk.Label, ttk.Entry]] = []
        self._axis_sel: Optional[str] = None
        self._vars: dict[str, tk.Variable] = {}
        self._wheel_widgets: list[tk.Misc] = []
        self._worker: Optional[WorkerProcess] = None
        self._job: dict = {}                       # 실행 중 작업 정보 (mode/out/stopping/error_shown/done)
        self._results: Optional[ResultList] = None
        self._result_dir: Optional[Path] = None
        self._orig_index: dict = {}
        self._ridx = 0
        self._results_mode = False
        self._build_ui()
        try:
            self.apply_settings(load_ui_state(self._state_file()))
        except Exception:
            # 타입이 잘못된 상태 파일: 기본값으로 시작 (종료 시 정상 상태로 덮어쓴다)
            self._keep_items = []
            self.apply_settings(BackgroundSwapSettings())
            self._status_lbl.configure(text="저장된 UI 상태를 읽을 수 없어 기본값으로 시작했습니다")
        self.bind("<Destroy>", self._on_destroy)
        self.winfo_toplevel().bind("<Escape>", self._cancel_pick, add="+")

    # 시험/교체용 이음매 (RunMixin이 사용)
    def _make_worker(self, args, on_event, on_exit) -> WorkerProcess:
        return WorkerProcess(args, on_event, on_exit)

    def _reset_output(self, output_dir: Path) -> None:
        reset_output_dir(output_dir)

    def _state_file(self) -> Path:
        return self._state_path or background_state.UI_STATE_PATH

    # ── Layout ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0, minsize=280)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_settings()
        self._build_preview()

    def _build_settings(self) -> None:
        outer = ttk.LabelFrame(self, text="Settings", padding=4)
        outer.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0, width=300, bg="#1e1e2e")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._canvas = canvas

        frame = ttk.Frame(canvas, padding=4)
        frame.columnconfigure(0, weight=1)
        frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(frame_id, width=e.width))

        self._build_data_group(frame).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._build_keep_group(frame).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._build_gen_group(frame).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self._axes_group = self._build_axes_group(frame)
        self._axes_group.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self._build_advanced(frame).grid(row=4, column=0, sticky="ew")

        self._bind_wheel(frame)
        self._bind_wheel(canvas)

    def _num_row(self, parent, row: int, attr: str, label: str, rng) -> None:
        var = tk.StringVar()
        self._vars[attr] = var
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=1)
        ttk.Spinbox(parent, from_=rng[0], to=rng[1], increment=rng[2], textvariable=var,
                    width=9).grid(row=row, column=1, sticky="w", pady=1)
        if attr in _PREVIEW_ATTRS or attr == "frame_stride":
            var.trace_add("write", lambda *a: self._schedule_refresh())

    def _path_row(self, parent, row: int, attr: str, label: str, browse) -> None:
        var = tk.StringVar()
        self._vars[attr] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, columnspan=2, sticky="w")
        entry = ttk.Entry(parent, textvariable=var, width=22)
        entry.grid(row=row + 1, column=0, sticky="ew")
        ttk.Button(parent, text="Browse", command=browse).grid(row=row + 1, column=1, padx=(4, 0))
        if attr in ("images_dir", "labels_dir"):
            entry.bind("<Return>", lambda e: self._on_dirs_changed())
            entry.bind("<FocusOut>", lambda e: self._on_dirs_changed())
        elif attr == "output_dir":
            entry.bind("<Return>", lambda e: self._on_output_changed())
            entry.bind("<FocusOut>", lambda e: self._on_output_changed())

    def _build_data_group(self, parent) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text="데이터", padding=4)
        g.columnconfigure(0, weight=1)
        self._path_row(g, 0, "images_dir", "Images folder:", lambda: self._browse_dir("images_dir"))
        self._path_row(g, 2, "labels_dir", "Labels folder:", lambda: self._browse_dir("labels_dir"))
        self._path_row(g, 4, "class_names_file", "Classes file:", self._browse_classes)
        self._path_row(g, 6, "output_dir", "Output folder:", lambda: self._browse_dir("output_dir"))
        sf = ttk.Frame(g)
        sf.grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))
        attr, _, label, rng = _DATA_NUM[0]
        self._num_row(sf, 0, attr, label, rng)
        return g

    def _build_keep_group(self, parent) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text="유지할 배경", padding=4)
        g.columnconfigure(0, weight=1)
        tree = ttk.Treeview(g, columns=("name", "method", "required"), show="headings",
                            height=4, selectmode="browse")
        for col, text, w in (("name", "이름", 120), ("method", "방식", 55), ("required", "필수", 40)):
            tree.heading(col, text=text)
            tree.column(col, width=w, anchor="w" if col == "name" else "center", stretch=col == "name")
        tree.grid(row=0, column=0, columnspan=3, sticky="ew")
        tree.bind("<Double-1>", lambda e: self._edit_keep())
        self._keep_tree = tree
        for c, (text, cmd) in enumerate((("추가", self._add_keep), ("편집", self._edit_keep),
                                         ("삭제", self._delete_keep))):
            ttk.Button(g, text=text, command=cmd, width=6).grid(row=1, column=c, pady=(4, 0))
        return g

    def _build_gen_group(self, parent) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text="생성", padding=4)
        for r, (attr, _, label, rng) in enumerate(_GEN_NUM):
            self._num_row(g, r, attr, label, rng)
        self._vars["use_llm"] = tk.BooleanVar()
        self._vars["use_vlm"] = tk.BooleanVar()
        ttk.Checkbutton(g, text="Use LLM (장면 프롬프트 생성)", variable=self._vars["use_llm"]).grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(g, text="Use VLM (결과 검수)", variable=self._vars["use_vlm"]).grid(
            row=4, column=0, columnspan=2, sticky="w")
        return g

    def _build_axes_group(self, parent) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text="장면 축 (값은 ; 로 구분)", padding=4)
        g.columnconfigure(0, weight=1)
        self._axes_body = ttk.Frame(g)
        self._axes_body.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._axes_body.columnconfigure(0, weight=1)
        ttk.Button(g, text="축 추가", command=self._add_axis).grid(row=1, column=0, pady=(4, 0))
        ttk.Button(g, text="축 삭제", command=self._delete_axis).grid(row=1, column=1, pady=(4, 0))
        return g

    def _build_advanced(self, parent) -> ttk.Frame:
        wrap = ttk.Frame(parent)
        wrap.columnconfigure(0, weight=1)
        self._adv_btn = ttk.Button(wrap, text="▶ 고급 설정", command=self._toggle_advanced)
        self._adv_btn.grid(row=0, column=0, sticky="ew")
        self._adv_frame = ttk.LabelFrame(wrap, text="고급", padding=4)
        self._adv_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._adv_frame.columnconfigure(1, weight=1)
        g = self._adv_frame
        self._vars["resolution_mode"] = tk.StringVar()
        ttk.Label(g, text="resolution_mode:").grid(row=0, column=0, sticky="w", pady=1)
        ttk.Combobox(g, textvariable=self._vars["resolution_mode"], values=("auto", "manual"),
                     state="readonly", width=8).grid(row=0, column=1, sticky="w", pady=1)
        for r, (attr, _, label, rng) in enumerate(_ADV_NUM, 1):
            self._num_row(g, r, attr, label, rng)
        r0 = len(_ADV_NUM) + 1
        for r, (key, label) in enumerate(_MODEL_LABELS, r0):
            var = tk.StringVar()
            self._vars[f"model_{key}"] = var
            ttk.Label(g, text=f"{label}:").grid(row=r, column=0, sticky="w", pady=1)
            ttk.Entry(g, textvariable=var, width=24).grid(row=r, column=1, sticky="ew", pady=1)
        self._adv_frame.grid_remove()      # 기본 접힘
        self._adv_open = False
        return wrap

    def _toggle_advanced(self) -> None:
        self._adv_open = not self._adv_open
        if self._adv_open:
            self._adv_frame.grid()
            self._adv_btn.configure(text="▼ 고급 설정")
        else:
            self._adv_frame.grid_remove()
            self._adv_btn.configure(text="▶ 고급 설정")

    def _build_preview(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        grid = ttk.Frame(frame)
        grid.grid(row=0, column=0, sticky="nsew")
        for i in range(2):
            grid.columnconfigure(i, weight=1)
            grid.rowconfigure(i, weight=1)
        titles = ("원본 + bbox (클릭: 색 선택)", "마스크 오버레이 (빨강=core 주황=보호 초록=keep)",
                  "생성 결과", "차이")
        labels = []
        for i, title in enumerate(titles):
            cell = ttk.Frame(grid)
            cell.grid(row=i // 2, column=i % 2, padx=3, pady=2, sticky="n")
            ttk.Label(cell, text=title).pack()
            holder = tk.Frame(cell, width=_PREVIEW_SIZE[0], height=_PREVIEW_SIZE[1], bg=_PANEL_BG)
            holder.pack()
            holder.pack_propagate(False)
            lbl = ttk.Label(holder, background=_PANEL_BG, text=_PLACEHOLDER, anchor="center")
            lbl.pack(expand=True)
            lbl.image = None
            labels.append(lbl)
        self._orig_lbl, self._mask_lbl, self._result_lbl, self._diff_lbl = labels
        for lbl in (self._orig_lbl, self._mask_lbl):
            lbl.configure(text="이미지 없음", compound="center")
            lbl.bind("<Button-1>", self._on_preview_click)

        nav = ttk.Frame(frame)
        nav.grid(row=1, column=0, pady=(4, 0))
        ttk.Button(nav, text="◀ Prev", command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=8)
        self._nav_lbl = ttk.Label(nav, text="–")
        self._nav_lbl.pack(side=tk.LEFT)
        ttk.Button(nav, text="Next ▶", command=lambda: self._step(1)).pack(side=tk.LEFT, padx=8)
        self._back_btn = ttk.Button(nav, text="마스크 미리보기로", command=self._leave_results)
        self._back_btn.pack(side=tk.LEFT, padx=(16, 0))

        self._status_lbl = ttk.Label(frame, text="Images/Labels 폴더를 선택하세요")
        self._status_lbl.grid(row=2, column=0, pady=(2, 0))
        self._detail_lbl = ttk.Label(frame, text="", wraplength=2 * _PREVIEW_SIZE[0], justify="left")
        self._detail_lbl.grid(row=3, column=0, pady=(2, 0))
        self._build_run_controls(frame)

    def _build_run_controls(self, frame) -> None:
        bar = ttk.Frame(frame)
        bar.grid(row=4, column=0, pady=(6, 0))
        specs = (("preview", f"Preview N={PREVIEW_N}", self._on_preview), ("run", "▶ Run", self._on_run),
                 ("stop", "■ Stop", self._on_stop), ("report", "Report", self._on_report),
                 ("export", "Export", self._on_export))
        self._btns: dict[str, ttk.Button] = {}
        for key, text, cmd in specs:
            self._btns[key] = ttk.Button(bar, text=text, command=cmd)
            self._btns[key].pack(side=tk.LEFT, padx=4)
        self._btns["stop"].state(["disabled"])
        prog = ttk.Frame(frame)
        prog.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        prog.columnconfigure(1, weight=1)
        self._stage_lbl = ttk.Label(prog, text="단계: –", width=22)
        self._stage_lbl.grid(row=0, column=0, sticky="w")
        self._progress = ttk.Progressbar(prog, mode="determinate", maximum=1, value=0)
        self._progress.grid(row=0, column=1, sticky="ew")

    def _bind_wheel(self, widget) -> None:
        def scroll(delta):
            self._canvas.yview_scroll(delta, "units")
            return "break"
        if isinstance(widget, (ttk.Treeview, ttk.Combobox)):
            return
        widget.bind("<MouseWheel>", lambda e: scroll(int(-1 * (e.delta / 120)) or (-1 if e.delta > 0 else 1)),
                    add="+")
        widget.bind("<Button-4>", lambda e: scroll(-1), add="+")
        widget.bind("<Button-5>", lambda e: scroll(1), add="+")
        for child in widget.winfo_children():
            self._bind_wheel(child)

    # ── Settings <-> widgets ─────────────────────────────────────────

    def _num(self, attr: str, fallback):
        v = _to_number(self._vars[attr].get(), _NUM_CASTS[attr])
        return fallback if v is None else v

    def build_settings(self) -> BackgroundSwapSettings:
        """위젯 → 설정. 숫자 칸이 비숫자면 이전 유효값을 유지한다 (예외 없음)."""
        prev = self._last
        kw: dict = {attr: self._num(attr, getattr(prev, attr)) for attr in _NUM_CASTS}
        for attr in _PATH_ATTRS:
            text = self._vars[attr].get().strip()
            kw[attr] = Path(text) if text else None
        kw["use_llm"] = bool(self._vars["use_llm"].get())
        kw["use_vlm"] = bool(self._vars["use_vlm"].get())
        mode = self._vars["resolution_mode"].get()
        kw["resolution_mode"] = mode if mode in ("auto", "manual") else prev.resolution_mode
        kw["models"] = ModelIds(**{
            k: self._vars[f"model_{k}"].get().strip() or getattr(prev.models, k)
            for k, _ in _MODEL_LABELS})
        kw["keep_items"] = copy.deepcopy(self._keep_items)
        kw["scene_axes"] = {name: parse_axis_values(var.get()) for name, var, _, _ in self._axis_rows}
        kw["allow_non_train"] = prev.allow_non_train      # UI 노출 없음: 실행별 확인 다이얼로그로만 설정
        self._last = BackgroundSwapSettings(**kw)
        return copy.deepcopy(self._last)

    def apply_settings(self, s: BackgroundSwapSettings) -> None:
        """설정 → 위젯."""
        self._last = copy.deepcopy(s)
        for attr in _PATH_ATTRS:
            v = getattr(s, attr)
            self._vars[attr].set("" if v is None else str(v))
        for attr in _NUM_CASTS:
            self._vars[attr].set(str(getattr(s, attr)))
        self._vars["use_llm"].set(s.use_llm)
        self._vars["use_vlm"].set(s.use_vlm)
        self._vars["resolution_mode"].set(s.resolution_mode)
        for key, _ in _MODEL_LABELS:
            self._vars[f"model_{key}"].set(getattr(s.models, key))
        self._keep_items = copy.deepcopy(s.keep_items)
        self._refresh_keep_tree()
        self._rebuild_axes(s.scene_axes)
        self._on_dirs_changed()

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_dir(self, attr: str) -> None:
        path = filedialog.askdirectory(title=f"Select {attr}")
        if not path:
            return
        self._vars[attr].set(path)
        if attr in ("images_dir", "labels_dir"):
            self._on_dirs_changed()
            self._autofill_classes()
        elif attr == "output_dir":
            self._on_output_changed()

    def _browse_classes(self) -> None:
        path = filedialog.askopenfilename(
            title="Select class names file",
            filetypes=[("Class names", "*.txt *.yaml *.yml *.names"), ("All files", "*.*")])
        if path:
            self._vars["class_names_file"].set(path)

    def _autofill_classes(self) -> None:
        """Classes 칸이 비어 있을 때만 자동 탐색."""
        if self._vars["class_names_file"].get().strip():
            return
        img, lab = (self._vars[k].get().strip() for k in ("images_dir", "labels_dir"))
        if img and lab:
            found = find_class_names_file(Path(img), Path(lab))
            if found is not None:
                self._vars["class_names_file"].set(str(found))

    def _on_dirs_changed(self) -> None:
        self._reload_items()
        self._schedule_refresh()

    def _on_output_changed(self) -> None:
        """Output 폴더에 이전 결과(Manifest)가 있으면 바로 탐색할 수 있게 불러온다."""
        text = self._vars["output_dir"].get().strip()
        if not text or self.is_running() or Path(text) == self._result_dir:
            return
        if (Path(text) / "Manifest.jsonl").is_file():
            self.load_results(Path(text))

    # ── Keep items ───────────────────────────────────────────────────

    def _refresh_keep_tree(self) -> None:
        self._keep_tree.delete(*self._keep_tree.get_children())
        for i, k in enumerate(self._keep_items):
            self._keep_tree.insert("", "end", iid=str(i),
                                   values=(k.name, k.method, "✔" if k.required else ""))

    def _selected_keep(self) -> Optional[int]:
        sel = self._keep_tree.selection()
        return int(sel[0]) if sel else None

    def _open_keep_dialog(self, item: Optional[KeepItem], skip_idx: Optional[int]) -> Optional[KeepItem]:
        names = [k.name for i, k in enumerate(self._keep_items) if i != skip_idx]
        dlg = KeepItemDialog(self, item, names)
        self.wait_window(dlg)
        return dlg.result

    def _add_keep(self) -> None:
        result = self._open_keep_dialog(None, None)
        if result is not None:
            self._keep_items.append(result)
            self._keep_changed()

    def _edit_keep(self) -> None:
        idx = self._selected_keep()
        if idx is None:
            return
        result = self._open_keep_dialog(self._keep_items[idx], idx)
        if result is not None:
            self._keep_items[idx] = result
            self._keep_changed()

    def _delete_keep(self) -> None:
        idx = self._selected_keep()
        if idx is not None:
            del self._keep_items[idx]
            self._keep_changed()

    def _keep_changed(self) -> None:
        self._refresh_keep_tree()
        self._schedule_refresh()

    # ── Scene axes ───────────────────────────────────────────────────

    def _rebuild_axes(self, axes: dict[str, list[str]]) -> None:
        for _, _, lbl, ent in self._axis_rows:
            lbl.destroy()
            ent.destroy()
        self._axis_rows, self._axis_sel = [], None
        for name, values in axes.items():
            self._append_axis_row(name, format_axis_values(values))

    def _append_axis_row(self, name: str, text: str) -> None:
        r = len(self._axis_rows)
        var = tk.StringVar(value=text)
        lbl = ttk.Label(self._axes_body, text=f"{name}:")
        lbl.grid(row=2 * r, column=0, sticky="w")
        ent = ttk.Entry(self._axes_body, textvariable=var, width=26)
        ent.grid(row=2 * r + 1, column=0, sticky="ew", pady=(0, 3))
        ent.bind("<FocusIn>", lambda e, n=name: self._select_axis(n))
        self._axis_rows.append((name, var, lbl, ent))
        self._bind_wheel(lbl)
        self._bind_wheel(ent)

    def _select_axis(self, name: str) -> None:
        self._axis_sel = name
        for n, _, lbl, _ in self._axis_rows:
            lbl.configure(foreground="#7c5cfc" if n == name else "#cdd6f4")

    def _add_axis(self) -> None:
        name = simpledialog.askstring("축 추가", "축 이름:", parent=self)
        name = (name or "").strip()
        if not name:
            return
        if any(n == name for n, _, _, _ in self._axis_rows):
            messagebox.showerror("Error", f"이미 있는 축입니다: {name}", parent=self)
            return
        self._append_axis_row(name, "")

    def _delete_axis(self) -> None:
        """값 칸을 마지막으로 클릭한 축을 삭제 (선택이 없으면 안내)."""
        if self._axis_sel is None:
            messagebox.showinfo("축 삭제", "삭제할 축의 값 칸을 먼저 클릭하세요.", parent=self)
            return
        rest = {n: parse_axis_values(v.get()) for n, v, _, _ in self._axis_rows if n != self._axis_sel}
        self._rebuild_axes(rest)

    # ── Items / preview ──────────────────────────────────────────────

    def _reload_items(self) -> None:
        img, lab = (self._vars[k].get().strip() for k in ("images_dir", "labels_dir"))
        stride = max(1, self._num("frame_stride", self._last.frame_stride))
        key = (img, lab, stride)
        if key == self._items_key:
            return
        self._items_key = key
        self._items, self._idx = [], 0
        if img and lab and Path(img).is_dir() and Path(lab).is_dir():
            try:
                self._items, _ = collect_items(Path(img), Path(lab), stride)
            except Exception:
                self._items = []

    def _schedule_refresh(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(_DEBOUNCE_MS, self._refresh_preview)

    def refresh_preview_now(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._refresh_preview()

    def has_preview_image(self) -> bool:
        return self._cur_img is not None

    def _step(self, d: int) -> None:
        if self._results_mode:
            n = len(self._results.entries)
            self._ridx = (self._ridx + d) % n
            self._show_result()
        elif self._items:
            self._idx = (self._idx + d) % len(self._items)
            self.refresh_preview_now()

    def _show_none(self, text: str) -> None:
        self._cur_img = None
        for lbl in (self._orig_lbl, self._mask_lbl):
            lbl.configure(image="", text="이미지 없음")
            lbl.image = None
        self._nav_lbl.configure(text="–")
        self._status_lbl.configure(text=text)

    def _refresh_preview(self) -> None:
        self._after_id = None
        if self._results_mode:
            return          # 결과 탐색 중에는 마스크 미리보기가 패널을 덮어쓰지 않는다
        self._reload_items()
        if not self._items:
            self._show_none("Images/Labels 폴더를 선택하세요 (이미지+라벨 쌍이 없습니다)")
            return
        item = self._items[self._idx]
        img = load_image_bgr(item.image_path)
        if img is None:
            self._show_none(f"이미지를 읽을 수 없습니다: {item.image_path.name}")
            return
        self._cur_img = img
        boxes = read_labels(item.label_path)
        overlay = quick_mask_preview(img, boxes, self.build_settings())
        for lbl, bgr in ((self._orig_lbl, draw_boxes(img, boxes)), (self._mask_lbl, overlay)):
            photo = self._to_photoimage(bgr)
            lbl.configure(image=photo, text="")
            lbl.image = photo
        self._nav_lbl.configure(text=f"{self._idx + 1} / {len(self._items)}")
        self._status_lbl.configure(text=item.image_path.name)

    def _to_photoimage(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize(_PREVIEW_SIZE, Image.LANCZOS)
        return ImageTk.PhotoImage(pil, master=self)

    # ── Click-to-pick ────────────────────────────────────────────────

    def begin_pick(self, on_pick: Callable, on_done: Callable) -> None:
        """다음 미리보기 클릭 1회를 on_pick(img_bgr, x, y)로 전달한다 (Esc 로 취소)."""
        self._pick = (on_pick, on_done)
        self._status_lbl.configure(text="미리보기에서 유지할 색 위치를 클릭하세요 (Esc 취소)")

    def _cancel_pick(self, event=None) -> None:
        if self._pick is not None:
            _, on_done = self._pick
            self._pick = None
            on_done()

    def _on_preview_click(self, event) -> None:
        if self._pick is None or self._cur_img is None:
            return
        lbl = event.widget
        pw, ph = _PREVIEW_SIZE
        px = event.x - max(0, (lbl.winfo_width() - pw) // 2)
        py = event.y - max(0, (lbl.winfo_height() - ph) // 2)
        if not (0 <= px < pw and 0 <= py < ph):
            return
        h, w = self._cur_img.shape[:2]
        x = min(max(int(px * w / pw), 0), w - 1)
        y = min(max(int(py * h / ph), 0), h - 1)
        on_pick, on_done = self._pick
        self._pick = None
        on_pick(self._cur_img, x, y)
        on_done()
        if self._items:
            self._status_lbl.configure(text=self._items[self._idx].image_path.name)

    # ── Results ─────────────────────────────────────────────

    def load_results(self, result_dir: Path) -> bool:
        """result_dir(Manifest/Accepted/Rejected/masks)의 결과를 4패널 탐색 모드로 연다."""
        rl = build_result_list(result_dir)
        self._results, self._result_dir, self._ridx = rl, Path(result_dir), 0
        images, labels = (self._vars[k].get().strip() for k in ("images_dir", "labels_dir"))
        try:
            self._orig_index = index_originals(Path(images), Path(labels)) if images and labels else {}
        except Exception:
            self._orig_index = {}
        if not rl.entries:
            self._leave_results()
            self._status_lbl.configure(text=f"표시할 결과가 없습니다 ({rl.summary()})")
            return False
        self._results_mode = True
        self._cur_img = None
        self._show_result()
        return True

    def _leave_results(self) -> None:
        was = self._results_mode
        self._results_mode = False
        for lbl in (self._result_lbl, self._diff_lbl):
            self._set_panel(lbl, None, _PLACEHOLDER)
        self._detail_lbl.configure(text="")
        if was or self._cur_img is None:
            self.refresh_preview_now()

    def _set_panel(self, lbl: ttk.Label, bgr: Optional[np.ndarray], text: str = "") -> None:
        if bgr is None:
            lbl.configure(image="", text=text)
            lbl.image = None
            return
        photo = self._to_photoimage(bgr)
        lbl.configure(image=photo, text="")
        lbl.image = photo

    def _show_result(self) -> None:
        rl = self._results
        e = rl.entries[self._ridx]
        self._nav_lbl.configure(text=f"{self._ridx + 1} / {len(rl.entries)}")
        self._status_lbl.configure(text=f"{e.stem}  v{e.variant} a{e.attempt} [{e.status}]  ·  {rl.summary()}")
        self._detail_lbl.configure(text=describe_record(e.record))
        try:
            item = self._orig_index.get(e.stem)
            if item is None:
                raise FileNotFoundError("원본 이미지를 찾을 수 없습니다 (Images/Labels 폴더를 확인하세요)")
            p = load_panels(self._result_dir, e, item)
        except Exception as ex:      # 파일 없음/손상: 패널에 사유만 표시
            for lbl in (self._orig_lbl, self._mask_lbl, self._result_lbl, self._diff_lbl):
                self._set_panel(lbl, None, str(ex)[:80])
            return
        for lbl, img in ((self._orig_lbl, p.orig), (self._mask_lbl, p.overlay),
                         (self._result_lbl, p.result), (self._diff_lbl, p.diff)):
            self._set_panel(lbl, img)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        if self._worker is not None:
            self._worker.stop()
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        try:
            save_ui_state(self.build_settings(), self._state_file())
        except Exception:
            pass
