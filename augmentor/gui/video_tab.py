import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.video import VideoInfo, get_video_info, convert_to_mp4, extract_frames


_MODES = ["WEBM → MP4 변환", "동영상 → 프레임 추출", "WEBM → MP4 → 프레임 추출 (연속)", "동영상 정보 보기"]


class VideoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._running = False
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)

        # Input
        inp_frame = ttk.LabelFrame(self, text="Input", padding=8)
        inp_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        inp_frame.columnconfigure(1, weight=1)

        ttk.Label(inp_frame, text="입력 파일:").grid(row=0, column=0, sticky="w")
        self._input_var = tk.StringVar()
        ttk.Entry(inp_frame, textvariable=self._input_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(inp_frame, text="Browse", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(inp_frame, text="출력 경로:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._output_var = tk.StringVar()
        ttk.Entry(inp_frame, textvariable=self._output_var).grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(inp_frame, text="Browse", command=self._browse_output).grid(row=1, column=2, pady=(4, 0))

        # Mode selection
        mode_frame = ttk.LabelFrame(self, text="작업 선택", padding=8)
        mode_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._mode_var = tk.IntVar(value=0)
        for i, label in enumerate(_MODES):
            rb = ttk.Radiobutton(mode_frame, text=label, variable=self._mode_var, value=i,
                                 command=self._on_mode_change)
            rb.pack(anchor="w")

        # Extract options
        self._extract_frame = ttk.LabelFrame(self, text="프레임 추출 옵션", padding=8)
        self._extract_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)

        ttk.Label(self._extract_frame, text="추출 비율:").pack(side=tk.LEFT)
        self._ratio_var = tk.IntVar(value=1)
        for val, label in [(1, "전체"), (2, "1/2"), (3, "1/3")]:
            ttk.Radiobutton(self._extract_frame, text=label, variable=self._ratio_var, value=val).pack(
                side=tk.LEFT, padx=4)

        self._gray_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._extract_frame, text="그레이스케일", variable=self._gray_var).pack(
            side=tk.LEFT, padx=8)

        # Info display
        self._info_frame = ttk.LabelFrame(self, text="동영상 정보", padding=8)
        self._info_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self._info_text = tk.Text(self._info_frame, height=7, state=tk.DISABLED, font=("Courier", 10))
        self._info_text.pack(fill=tk.X)

        # Progress + run
        bottom = ttk.Frame(self)
        bottom.grid(row=4, column=0, sticky="ew", padx=8, pady=8)
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(bottom, variable=self._progress_var, maximum=100).grid(
            row=0, column=0, sticky="ew")

        self._status_lbl = ttk.Label(bottom, text="Ready")
        self._status_lbl.grid(row=1, column=0)

        self._run_btn = ttk.Button(bottom, text="▶ Run", command=self._on_run)
        self._run_btn.grid(row=2, column=0, pady=4)

        self._on_mode_change()  # set initial visibility

    # ── File browsing ────────────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.webm *.mp4 *.avi *.mkv *.mov"), ("All", "*.*")],
        )
        if not path:
            return
        self._input_var.set(path)
        inp = Path(path)
        mode = self._mode_var.get()
        if mode == 0:
            self._output_var.set(str(inp.with_suffix(".mp4")))
        elif mode in (1, 2):
            self._output_var.set(str(inp.parent / (inp.stem + "_frames")))

    def _browse_output(self):
        mode = self._mode_var.get()
        if mode in (1, 2):
            path = filedialog.askdirectory(title="Select output directory")
        else:
            path = filedialog.asksaveasfilename(
                title="Save MP4 as",
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4")],
            )
        if path:
            self._output_var.set(path)

    # ── Mode switch ──────────────────────────────────────────────────

    def _on_mode_change(self):
        mode = self._mode_var.get()
        if mode in (1, 2):
            self._extract_frame.grid()
        else:
            self._extract_frame.grid_remove()
        if mode == 3:
            self._info_frame.grid()
        else:
            self._info_frame.grid_remove()

    # ── Run ─────────────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        mode = self._mode_var.get()
        inp = self._input_var.get().strip()
        if not inp:
            messagebox.showerror("Error", "입력 파일을 선택하세요.")
            return

        self._running = True
        self._run_btn.configure(state=tk.DISABLED)
        self._progress_var.set(0)
        self._status_lbl.configure(text="Running…")

        thread = threading.Thread(target=self._run_worker, args=(mode, inp), daemon=True)
        thread.start()

    def _run_worker(self, mode: int, inp: str):
        try:
            inp_path = Path(inp)
            out = self._output_var.get().strip()

            if mode == 0:
                self._do_convert(inp_path, Path(out))
            elif mode == 1:
                self._do_extract(inp_path, Path(out))
            elif mode == 2:
                mp4_path = inp_path.with_suffix(".mp4")
                self._do_convert(inp_path, mp4_path)
                self._do_extract(mp4_path, Path(out))
            elif mode == 3:
                self._do_info(inp_path)

            self.after(0, self._on_complete)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _do_convert(self, inp: Path, out: Path):
        self.after(0, self._status_lbl.configure, {"text": "변환 중…"})
        convert_to_mp4(inp, out, progress_cb=self._on_convert_progress)

    def _do_extract(self, inp: Path, out: Path):
        self.after(0, self._status_lbl.configure, {"text": "프레임 추출 중…"})
        saved = extract_frames(
            inp, out,
            ratio=self._ratio_var.get(),
            grayscale=self._gray_var.get(),
            progress_cb=self._on_extract_progress,
        )
        self.after(0, self._status_lbl.configure, {"text": f"추출 완료: {saved}장"})

    def _do_info(self, inp: Path):
        info = get_video_info(inp)
        self.after(0, self._display_info, info)

    def _display_info(self, info: VideoInfo):
        lines = []
        if not info.is_valid():
            lines.append(f"오류: {info.error}")
        else:
            lines.append(f"포맷:      {info.format}")
            mins = int(info.duration_sec) // 60
            secs = int(info.duration_sec) % 60
            lines.append(f"길이:      {mins}분 {secs}초")
            if info.bitrate_bps:
                lines.append(f"비트레이트: {info.bitrate_bps // 1000} kbps")
            if info.video_codec:
                lines.append(f"비디오:    {info.video_codec}  {info.width}×{info.height}  {info.fps:.2f} fps")
            if info.audio_codec:
                lines.append(f"오디오:    {info.audio_codec}  {info.sample_rate} Hz  {info.channels}ch")

        self._info_text.configure(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, "\n".join(lines))
        self._info_text.configure(state=tk.DISABLED)

    def _on_convert_progress(self, pct: int):
        self.after(0, self._progress_var.set, float(pct))

    def _on_extract_progress(self, done: int, total: int):
        if total > 0:
            pct = done / total * 100
        else:
            pct = 100.0
        self.after(0, self._progress_var.set, pct)

    def _on_complete(self):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._progress_var.set(100)
        self._status_lbl.configure(text="완료!")

    def _on_error(self, msg: str):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL)
        self._status_lbl.configure(text="오류 발생")
        messagebox.showerror("오류", msg)
