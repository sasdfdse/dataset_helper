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
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from background_swap.dataset.classes import find_class_names_file
from background_swap.stages.composite import draw_boxes
from background_swap.dataset.inputs import collect_items, load_image_bgr
from background_swap.common.settings import (DEFAULT_SCENE_AXES, MODEL_PRESETS, PRESET_USE_VLM,
                                             BackgroundSwapSettings, KeepItem, ModelIds, axis_label,
                                             is_sdxl, model_preset)
from background_swap.common.estimate import describe_plan, make_plan
from background_swap.common.hardware import (PROFILE_PATH, HardwareProfile, load_profile, merge_probe,
                                             save_profile)
from background_swap.common.store import reset_output_dir
from core.label import read_labels
from gui import background_state
from gui.background_state import (format_axis_values, load_ui_state,
                                  parse_axis_values, quick_mask_preview,
                                  save_ui_state, to_number as _to_number)
from gui.background_results import (ResultList, build_result_list, describe_record,
                                    index_originals, load_panels, originals_from_items)
from gui.background_dialogs import ReportWindow
from gui.background_run import PREVIEW_N, RunMixin
from gui.keep_item_dialog import KeepItemDialog
from gui.info_texts import BACKGROUND_INFO
from gui.placeholder import Placeholder
from gui.worker_process import WorkerProcess
from gui import app_prefs, rounded, theme

_PLACEHOLDER = "Preview 실행 후 표시"
_DEBOUNCE_MS = 150
_PANEL_BG = theme.ENTRY
_PREVIEW_SIZE = (340, 215)  # fits 4 panels plus details in one window
_MODEL_LABELS = (("segmenter", "Segmenter"), ("detector", "Detector"), ("llm", "LLM"),
                 ("inpainter", "Inpainter"), ("vlm", "VLM"))

# (settings field, type, label, spinbox (from, to, increment) or None)
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
_IMAGE_TYPES = [("이미지", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG"), ("모든 파일", "*")]
FAST_MODE_LABEL = "빠른 모드 (약 2.7배 빠름)"
# numeric fields that trigger re-estimation (frame_stride re-estimates via the preview refresh)
_ESTIMATE_ATTRS = ("num_variants", "max_attempts", "max_resolution_px", "manual_resolution_px")




class _AxisRow:
    """배경 조합 항목 한 줄: 체크박스(사용 여부) + 참고 사진 버튼(선택) + 후보 값 칸."""

    def __init__(self, name: str, values: tk.StringVar, enabled: tk.BooleanVar,
                 refs: Optional[list[Path]] = None) -> None:
        self.name, self.values, self.enabled = name, values, enabled
        self.refs: list[Path] = list(refs or [])
        self.check: Optional[ttk.Checkbutton] = None
        self.entry: Optional[ttk.Entry] = None
        self.placeholder: Optional[Placeholder] = None
        self.ref_btn: Optional[ttk.Button] = None
        self.ref_clear: Optional[ttk.Button] = None
        self.widgets: list[tk.Misc] = []

class BackgroundTab(RunMixin, ttk.Frame):
    def __init__(self, parent, state_path: Optional[Path] = None):
        super().__init__(parent)
        self._state_path = state_path
        self._last = BackgroundSwapSettings()      # last valid values (fallback for invalid input)
        self._keep_items: list[KeepItem] = []
        self._items = []                           # list[ImageItem]
        self._items_key = None
        self._idx = 0
        self._cur_img: Optional[np.ndarray] = None
        self._after_id: Optional[str] = None
        self._pick: Optional[tuple[Callable, Callable]] = None
        self._axis_rows: list[_AxisRow] = []
        self._vars: dict[str, tk.Variable] = {}
        self._wheel_widgets: list[tk.Misc] = []
        self._worker: Optional[WorkerProcess] = None
        self._job: dict = {}                       # running job info (mode/out/stopping/error/done)
        self._results: Optional[ResultList] = None
        self._result_dir: Optional[Path] = None
        self._orig_index: dict = {}
        self._ridx = 0
        self._results_mode = False
        self._estimate_after: Optional[str] = None
        self._last_job_note = ""
        # GPU profile: start from the saved value; start_hardware_probe() re-measures
        self._hw_path = self._state_file().parent / PROFILE_PATH.name
        self._hw = load_profile(self._hw_path)
        self._probe: Optional[WorkerProcess] = None
        rounded.install(self.winfo_toplevel())
        self._build_ui()
        try:
            self.apply_settings(load_ui_state(self._state_file()))
        except Exception:
            # state file with wrong types: start from defaults (overwritten with a valid state on exit)
            self._keep_items = []
            self.apply_settings(BackgroundSwapSettings())
            self.refresh_preview_now()  # flush the pending refresh so it doesn't overwrite the notice below
            self._status_lbl.configure(text="저장된 UI 상태를 읽을 수 없어 기본값으로 시작했습니다")
        self.bind("<Destroy>", self._on_destroy)
        self.winfo_toplevel().bind("<Escape>", self._cancel_pick, add="+")

    # seam for tests (used by RunMixin)
    def _make_worker(self, args, on_event, on_exit) -> WorkerProcess:
        return WorkerProcess(args, on_event, on_exit)

    def _reset_output(self, output_dir: Path) -> None:
        reset_output_dir(output_dir)

    def _state_file(self) -> Path:
        return self._state_path or background_state.UI_STATE_PATH

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

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0, width=300, bg=theme.BG)
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
        if attr in _ESTIMATE_ATTRS:
            var.trace_add("write", lambda *a: self._schedule_estimate())

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
            ttk.Button(g, text=text, command=cmd, width=6).grid(row=1, column=c, padx=2, pady=(4, 0))
        return g

    def _build_gen_group(self, parent) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text="생성", padding=4)
        for r, (attr, _, label, rng) in enumerate(_GEN_NUM):
            self._num_row(g, r, attr, label, rng)
        self._vars["use_llm"] = tk.BooleanVar()
        self._vars["use_vlm"] = tk.BooleanVar()
        ttk.Checkbutton(g, text="Use LLM (배경 문장 생성)", variable=self._vars["use_llm"]).grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(g, text="Use VLM (결과 검수)", variable=self._vars["use_vlm"]).grid(
            row=4, column=0, columnspan=2, sticky="w")
        self._vars["fast_mode"] = tk.BooleanVar()
        self._fast_check = ttk.Checkbutton(g, text=FAST_MODE_LABEL, variable=self._vars["fast_mode"])
        self._fast_check.grid(row=5, column=0, columnspan=2, sticky="w")
        # picking a size also updates the model names and VLM setting in advanced settings
        size = ttk.Frame(g)
        size.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(size, text="모델:").grid(row=0, column=0, sticky="nw")
        self._preset_var = tk.StringVar(value="large")
        for r, (key, text) in enumerate((("large", "Large (고품질)"), ("small", "Small (저사양 PC용)"))):
            ttk.Radiobutton(size, text=text, value=key, variable=self._preset_var,
                            command=lambda k=key: self._apply_preset(k)).grid(row=r, column=1, sticky="w",
                                                                              padx=(6, 0))
        self._preset_note = ttk.Label(g, text="", foreground=theme.FG_DIM, wraplength=230, justify="left")
        self._preset_note.grid(row=7, column=0, columnspan=2, sticky="w")
        for key in ("use_llm", "use_vlm", "fast_mode"):
            self._vars[key].trace_add("write", lambda *a: self._schedule_estimate())
        return g

    def _build_axes_group(self, parent) -> ttk.LabelFrame:
        g = ttk.LabelFrame(parent, text="배경 조합 (체크한 항목만 사용)", padding=4)
        g.columnconfigure(0, weight=1)
        hint = ttk.Label(g, text="새 배경마다 항목별로 후보 하나를 무작위로 골라 조합한다. "
                         "후보는 ; 로 구분. 비워 두면 회색 예시 값을 쓴다. "
                         "📷 사진: 참고 사진(선택)을 고르면 그 항목의 후보에 사진 설명이 더해진다.",
                         foreground=theme.FG_DIM, wraplength=230, justify="left")
        hint.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self._axes_body = ttk.Frame(g)
        self._axes_body.grid(row=1, column=0, sticky="ew")
        self._axes_body.columnconfigure(0, weight=1)
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
        self._vars["resolution_mode"].trace_add("write", lambda *a: self._schedule_estimate())
        ttk.Label(g, text="resolution_mode:").grid(row=0, column=0, sticky="w", pady=1)
        ttk.Combobox(g, textvariable=self._vars["resolution_mode"], values=("auto", "manual"),
                     state="readonly", width=8).grid(row=0, column=1, sticky="w", pady=1)
        for r, (attr, _, label, rng) in enumerate(_ADV_NUM, 1):
            self._num_row(g, r, attr, label, rng)
        r0 = len(_ADV_NUM) + 1
        for r, (key, label) in enumerate(_MODEL_LABELS, r0):
            var = tk.StringVar()
            self._vars[f"model_{key}"] = var
            var.trace_add("write", lambda *a: self._sync_preset())
            ttk.Label(g, text=f"{label}:").grid(row=r, column=0, sticky="w", pady=1)
            ttk.Entry(g, textvariable=var, width=24).grid(row=r, column=1, sticky="ew", pady=1)
        self._adv_frame.grid_remove()
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
        self._titles = []
        for i, title in enumerate(titles):
            cell = ttk.Frame(grid)
            cell.grid(row=i // 2, column=i % 2, padx=3, pady=2, sticky="n")
            tl = ttk.Label(cell, text=title)
            tl.pack()
            self._titles.append((tl, title))
            holder = tk.Frame(cell, width=_PREVIEW_SIZE[0], height=_PREVIEW_SIZE[1], bg=theme.BG)
            holder.pack()
            holder.pack_propagate(False)
            lbl = ttk.Label(holder, style=rounded.PANEL_STYLE, text=_PLACEHOLDER, anchor="center")
            lbl.pack(fill=tk.BOTH, expand=True)
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
        bottom = ttk.Frame(frame)
        bottom.grid(row=6, column=0, sticky="ew", pady=(4, 0))
        bottom.columnconfigure(0, weight=1)
        self._eta_lbl = ttk.Label(bottom, text="", foreground=theme.FG_DIM,
                                  wraplength=2 * _PREVIEW_SIZE[0] - 60, justify="left")
        self._eta_lbl.grid(row=0, column=0, sticky="w")
        self._info_btn = ttk.Button(bottom, text="ⓘ Info", width=7, command=self._on_info)
        self._info_btn.grid(row=0, column=1, sticky="e", padx=(6, 0))

    def _schedule_estimate(self) -> None:
        if self._estimate_after is not None:
            try:
                self.after_cancel(self._estimate_after)
            except tk.TclError:
                pass
        self._estimate_after = self.after(_DEBOUNCE_MS, self._update_estimate)

    def _update_estimate(self) -> None:
        """실행 전 예상 시간. 실행 중에는 진행 이벤트가 남은 시간을 표시하므로 건드리지 않는다."""
        self._estimate_after = None
        if self.is_running():
            return
        self._reload_items()
        s = self.build_settings()
        size = (self._items[0].width, self._items[0].height) if self._items else None
        text = describe_plan(make_plan(s, len(self._items), size, self._hw), s)
        if self._last_job_note:
            text += "\n" + self._last_job_note
        self._eta_lbl.configure(text=text)

    def _current_models(self) -> ModelIds:
        return ModelIds(**{k: self._vars[f"model_{k}"].get().strip() for k, _ in _MODEL_LABELS})

    def _apply_preset(self, name: str) -> None:
        """Small/Large: 모델 이름과 VLM 사용 여부를 프리셋 값으로 바꾼다 (Small은 빠른 모드도 끈다)."""
        preset = MODEL_PRESETS[name]
        for key, _ in _MODEL_LABELS:
            self._vars[f"model_{key}"].set(getattr(preset, key))
        self._vars["use_vlm"].set(PRESET_USE_VLM[name])
        if name == "small":
            self._vars["fast_mode"].set(False)
        self._sync_preset()

    def _sync_preset(self) -> None:
        """모델 이름 칸에 맞춰 Small/Large 표시와 빠른 모드 사용 가능 여부를 갱신한다."""
        if not hasattr(self, "_preset_var"):
            return
        name = model_preset(self._current_models())
        self._preset_var.set(name if name != "custom" else "")
        sdxl = is_sdxl(self._vars["model_inpainter"].get())
        self._fast_check.state(["!disabled"] if sdxl else ["disabled"])
        self._preset_note.configure(text={
            "large": "Large: SDXL 인페인팅, 8GB GPU 기준. make venv 설치.",
            "small": "Small: SD1.5 인페인팅(512px), 작은 LLM, VLM 검수 끔. 4GB급 GPU에서도 동작. "
                     "make venv-small 설치로도 충분.",
            "custom": "모델: 고급 설정에서 직접 지정함",
        }[name])
        self._schedule_estimate()

    def _on_info(self) -> None:
        text = BACKGROUND_INFO + f"\n[이 PC]\n• {self._hw.describe()}\n"
        ReportWindow(self, text, title="Info — Background Swap", size="680x760")

    def start_hardware_probe(self) -> None:
        """앱 시작 때 한 번: 워커로 GPU 이름/VRAM/속도를 재서 이 PC에 맞춘다 (1~3초, 백그라운드)."""
        if self._probe is not None:
            return
        self._probe = self._make_worker(
            ["probe"], lambda ev: self._post(self._on_probe_event, ev),
            lambda code, err: self._post(self._on_probe_exit))
        try:
            self._probe.start()
        except OSError:
            self._probe = None

    def _on_probe_event(self, ev: dict) -> None:
        if ev.get("event") != "done":
            return
        gpu = ev.get("gpu")
        if not isinstance(gpu, str) or not gpu:
            self._hw = HardwareProfile()       # no CUDA GPU: use reference values
        else:
            try:
                self._hw = merge_probe(self._hw_path, gpu, float(ev.get("vram_gb") or 0),
                                       float(ev.get("tflops") or 0))
                save_profile(self._hw, self._hw_path)
            except (OSError, TypeError, ValueError):
                pass
        self._schedule_estimate()

    def _on_probe_exit(self) -> None:
        self._probe = None

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
        kw["fast_mode"] = bool(self._vars["fast_mode"].get())
        mode = self._vars["resolution_mode"].get()
        kw["resolution_mode"] = mode if mode in ("auto", "manual") else prev.resolution_mode
        kw["models"] = ModelIds(**{
            k: self._vars[f"model_{k}"].get().strip() or getattr(prev.models, k)
            for k, _ in _MODEL_LABELS})
        kw["keep_items"] = copy.deepcopy(self._keep_items)
        # empty field = the greyed-out example values;
        # with reference photos, an empty field uses only the photo captions (added at run time)
        kw["scene_axes"] = {row.name: parse_axis_values(row.values.get())
                            or ([] if row.refs else list(DEFAULT_SCENE_AXES.get(row.name, [])))
                            for row in self._axis_rows}
        kw["reference_images"] = {row.name: list(row.refs) for row in self._axis_rows if row.refs}
        kw["disabled_scene_axes"] = [row.name for row in self._axis_rows if not row.enabled.get()]
        kw["allow_non_train"] = prev.allow_non_train      # not in the UI; set only through the per-run confirmation dialog
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
        self._vars["fast_mode"].set(s.fast_mode)
        self._vars["resolution_mode"].set(s.resolution_mode)
        for key, _ in _MODEL_LABELS:
            self._vars[f"model_{key}"].set(getattr(s.models, key))
        self._keep_items = copy.deepcopy(s.keep_items)
        self._refresh_keep_tree()
        self._rebuild_axes(s.scene_axes, s.disabled_scene_axes, s.reference_images)
        self._on_dirs_changed()
        self._schedule_estimate()

    def _browse_dir(self, attr: str) -> None:
        path = app_prefs.ask_directory(f"Select {attr}", self._vars[attr].get()
                                       or self._vars["images_dir"].get())
        if not path:
            return
        self._vars[attr].set(path)
        if attr in ("images_dir", "labels_dir"):
            self._on_dirs_changed()
            self._autofill_classes()
        elif attr == "output_dir":
            self._on_output_changed()

    def _browse_classes(self) -> None:
        path = app_prefs.ask_open_file(
            "Select class names file", self._vars["class_names_file"].get() or self._vars["labels_dir"].get(),
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

    def _rebuild_axes(self, axes: dict[str, list[str]], disabled: list[str],
                      refs: Optional[dict[str, list[Path]]] = None) -> None:
        for row in self._axis_rows:
            for w in row.widgets:
                w.destroy()
        self._axis_rows = []
        for name, values in axes.items():
            self._append_axis_row(name, values, name not in disabled, (refs or {}).get(name, []))

    def _append_axis_row(self, name: str, values: list[str], enabled: bool,
                         refs: Optional[list[Path]] = None) -> None:
        r = len(self._axis_rows)
        example = DEFAULT_SCENE_AXES.get(name, [])
        # values equal to the example: leave the field empty and show the grey hint (empty = use example)
        text = "" if example and list(values) == example and not refs else format_axis_values(values)
        row = _AxisRow(name, tk.StringVar(value=text), tk.BooleanVar(value=enabled), refs)
        row.check = ttk.Checkbutton(self._axes_body, text=axis_label(name), variable=row.enabled,
                                    command=lambda: self._sync_axis_row(row))
        row.check.grid(row=2 * r, column=0, sticky="w")
        btns = ttk.Frame(self._axes_body)
        btns.grid(row=2 * r, column=1, sticky="e")
        row.ref_btn = ttk.Button(btns, width=9, command=lambda: self._pick_refs(row))
        row.ref_btn.pack(side=tk.LEFT)
        row.ref_clear = ttk.Button(btns, text="✕", width=2, command=lambda: self._clear_refs(row))
        row.entry = ttk.Entry(self._axes_body, textvariable=row.values, width=26)
        row.entry.grid(row=2 * r + 1, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        row.placeholder = Placeholder(row.entry, row.values, format_axis_values(example))
        row.widgets = [row.check, btns, row.entry]
        self._axis_rows.append(row)
        self._sync_axis_row(row)
        for w in (row.check, row.entry, btns, row.ref_btn, row.ref_clear):
            self._bind_wheel(w)

    def _pick_refs(self, row: "_AxisRow") -> None:
        files = app_prefs.ask_open_files(
            f"'{axis_label(row.name)}' 참고 사진 선택 (여러 장 가능)",
            str(row.refs[0]) if row.refs else "", filetypes=_IMAGE_TYPES)
        if files:
            row.refs = [Path(f) for f in files]
            self._sync_axis_row(row)
            self._schedule_estimate()

    def _clear_refs(self, row: "_AxisRow") -> None:
        row.refs = []
        self._sync_axis_row(row)
        self._schedule_estimate()

    @staticmethod
    def _sync_axis_row(row: "_AxisRow") -> None:
        """체크를 끈 항목은 값 칸과 사진 버튼을 흐리게(편집 불가) 한다. 값과 사진은 그대로 남는다."""
        state = ["!disabled"] if row.enabled.get() else ["disabled"]
        row.entry.state(state)
        if row.ref_btn is not None:
            row.ref_btn.configure(text=f"📷 {len(row.refs)}장" if row.refs else "📷 사진")
            row.ref_btn.state(state)
            row.ref_clear.state(state)
            if row.refs:
                row.ref_clear.pack(side=tk.LEFT, padx=(2, 0))
            else:
                row.ref_clear.pack_forget()
        if row.placeholder is not None:
            example = format_axis_values(DEFAULT_SCENE_AXES.get(row.name, []))
            row.placeholder.set_text("비워 두면 참고 사진 설명만 사용" if row.refs else example)

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
        self._schedule_estimate()
        if self._results_mode:
            return          # while browsing results, don't let the mask preview overwrite the panels
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
        return ImageTk.PhotoImage(rounded.round_corners(pil), master=self)

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

    def load_results(self, result_dir: Path) -> bool:
        """result_dir(Manifest/Accepted/Rejected/masks)의 결과를 4패널 탐색 모드로 연다."""
        rl = build_result_list(result_dir)
        self._results, self._result_dir, self._ridx = rl, Path(result_dir), 0
        # Originals come from that run's items.jsonl (the folder fields may point at another dataset).
        # Only fall back to the current Images/Labels folders when items.jsonl is missing.
        try:
            index = originals_from_items(Path(result_dir))
        except Exception:
            index = {}
        if index is None:
            images, labels = (self._vars[k].get().strip() for k in ("images_dir", "labels_dir"))
            try:
                index = index_originals(Path(images), Path(labels)) if images and labels else {}
            except Exception:
                index = {}
        self._orig_index = index
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
        self._set_diff_title("")
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

    def _set_diff_title(self, note: str) -> None:
        tl, base = self._titles[3]
        tl.configure(text=f"{base} {note}" if note else base)

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
        except Exception as ex:
            for lbl in (self._orig_lbl, self._mask_lbl, self._result_lbl, self._diff_lbl):
                self._set_panel(lbl, None, str(ex)[:80])
            self._set_diff_title("")
            return
        self._set_diff_title(p.diff_note)
        for lbl, img in ((self._orig_lbl, p.orig), (self._mask_lbl, p.overlay),
                         (self._result_lbl, p.result), (self._diff_lbl, p.diff)):
            self._set_panel(lbl, img)

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        for proc in (self._worker, self._probe):
            if proc is not None:
                proc.stop()
        for attr in ("_after_id", "_estimate_after"):
            if getattr(self, attr) is not None:
                try:
                    self.after_cancel(getattr(self, attr))
                except tk.TclError:
                    pass
                setattr(self, attr, None)
        try:
            save_ui_state(self.build_settings(), self._state_file())
        except Exception:
            pass
