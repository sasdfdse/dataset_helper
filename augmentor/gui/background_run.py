"""Background Swap 탭의 실행 제어: Run/Preview/Export 사전 점검, 워커 실행·중지, 이벤트 처리, Report.

BackgroundTab에 섞어 쓰는 mixin. 탭이 제공해야 하는 것: _vars, _btns, _status_lbl, _stage_lbl,
_progress, build_settings(), load_results(), 그리고 시험/교체용 이음매 _make_worker(), _reset_output().
"""
import dataclasses
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox

from background_swap.inputs import looks_non_train
from background_swap.settings import (BackgroundSwapSettings, generation_diff, load_settings,
                                      save_settings, validate_settings)
from gui.background_dialogs import ReportWindow, ask_choice
from gui.background_results import format_export_message, format_report, load_report

_STAGE_KO = {"inputs": "입력", "masks": "마스크", "prompts": "프롬프트", "inpaint": "인페인팅",
             "validate": "검증", "report": "리포트"}
_DEPS_HINT = "pip install -r requirements-bg.txt (augmentor/.venv 사용 권장)"
PREVIEW_N = 3


class RunMixin:
    def is_running(self) -> bool:
        return self._worker is not None

    def _checked_settings(self) -> Optional[BackgroundSwapSettings]:
        """설정 검증 + 학습용 폴더 가드. 실패/취소면 None (allow_non_train은 이번 실행에만 적용)."""
        s = self.build_settings()
        errors = validate_settings(s)
        if errors:
            messagebox.showerror("설정 오류", "\n".join(errors), parent=self)
            return None
        if not s.allow_non_train and looks_non_train(Path(s.images_dir)):
            if not messagebox.askyesno(
                    "학습용이 아닌 폴더?",
                    f"이미지 폴더가 val/test 등 학습용이 아닌 폴더로 보입니다:\n{s.images_dir}\n\n"
                    "그래도 이 폴더로 실행할까요?", parent=self):
                return None
            s = dataclasses.replace(s, allow_non_train=True)
        return s

    def _save_job_settings(self, s: BackgroundSwapSettings, path: Path) -> bool:
        try:
            save_settings(s, path)
            return True
        except OSError as e:
            messagebox.showerror("Error", f"설정을 저장할 수 없습니다: {path}\n{e}", parent=self)
            return False

    def _ask_mismatch(self, diff: list[str]) -> str:
        """'new' | 'overwrite' | 'cancel'"""
        return ask_choice(
            self, "기존 출력과 설정이 다릅니다",
            "출력 폴더에 다른 설정으로 만든 결과가 있습니다.\n달라진 항목: " + ", ".join(diff)
            + "\n\n새 출력 폴더를 쓰거나, 기존 결과를 지우고 덮어쓸 수 있습니다.",
            [("new", "새 출력 폴더"), ("overwrite", "덮어쓰기"), ("cancel", "취소")])

    @staticmethod
    def _output_diff(s: BackgroundSwapSettings) -> list[str]:
        existing = Path(s.output_dir) / "run_settings.json"
        if not existing.is_file():
            return []
        try:
            return generation_diff(load_settings(existing), s)
        except Exception:
            return ["run_settings.json (읽을 수 없음)"]

    def _resolve_output_conflict(self, s: BackgroundSwapSettings) -> Optional[BackgroundSwapSettings]:
        """기존 run_settings.json과 생성 설정이 다르면 3지선다. 진행 가능하면 설정, 아니면 None."""
        while True:
            diff = self._output_diff(s)
            if not diff:
                return s
            choice = self._ask_mismatch(diff)
            if choice == "overwrite":
                try:
                    self._reset_output(Path(s.output_dir))
                except OSError as e:
                    messagebox.showerror("Error", f"기존 출력을 지울 수 없습니다: {s.output_dir}\n{e}",
                                         parent=self)
                    return None
                return s
            if choice != "new":
                return None
            path = filedialog.askdirectory(title="새 출력 폴더 선택")
            if not path:
                return None
            s = dataclasses.replace(s, output_dir=Path(path))
            errors = validate_settings(s)
            if errors:
                messagebox.showerror("설정 오류", "\n".join(errors), parent=self)
                return None
            self._vars["output_dir"].set(path)

    def _on_run(self) -> None:
        if self.is_running():
            return
        s = self._checked_settings()
        s = self._resolve_output_conflict(s) if s is not None else None
        if s is None:
            return
        path = Path(s.output_dir) / "run_settings.json"
        if self._save_job_settings(s, path):
            self._launch("run", ["run", "--settings", str(path)], Path(s.output_dir))

    def _on_preview(self) -> None:
        if self.is_running():
            return
        s = self._checked_settings()
        if s is None:
            return
        # run_settings.json(이어하기 기준)은 건드리지 않도록 별도 파일에 저장한다
        path = Path(s.output_dir) / "_preview_settings.json"
        if self._save_job_settings(s, path):
            self._launch("preview", ["preview", "--settings", str(path), "--n", str(PREVIEW_N)],
                         Path(s.output_dir))

    def _on_export(self) -> None:
        if self.is_running():
            return
        out_text = self._vars["output_dir"].get().strip()
        run_settings = Path(out_text) / "run_settings.json" if out_text else None
        if run_settings is None or not run_settings.is_file():
            messagebox.showerror("Error", "내보낼 결과가 없습니다. 먼저 Run으로 결과를 만드세요 "
                                 "(Output 폴더에 run_settings.json 필요).", parent=self)
            return
        images = filedialog.askdirectory(title="내보낼 images 폴더 선택")
        if not images:
            return
        labels = filedialog.askdirectory(title="내보낼 labels 폴더 선택")
        if not labels:
            return
        settings_path = run_settings
        if looks_non_train(Path(images)) or looks_non_train(Path(labels)):
            if not messagebox.askyesno(
                    "학습용이 아닌 폴더?",
                    f"대상 폴더가 val/test 등 학습용이 아닌 폴더로 보입니다:\n{images}\n{labels}\n\n"
                    "그래도 내보낼까요?", parent=self):
                return
            try:
                allowed = dataclasses.replace(load_settings(run_settings), allow_non_train=True)
            except Exception as e:
                messagebox.showerror("Error", f"run_settings.json을 읽을 수 없습니다: {e}", parent=self)
                return
            settings_path = Path(out_text) / "_export_settings.json"
            if not self._save_job_settings(allowed, settings_path):
                return
        self._launch("export", ["export", "--settings", str(settings_path),
                                "--images-dir", images, "--labels-dir", labels], Path(out_text))

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._job["stopping"] = True
            self._status_lbl.configure(text="중지하는 중…")
            self._worker.stop()

    def _launch(self, mode: str, args: list[str], out: Path) -> None:
        self._job = {"mode": mode, "out": out, "stopping": False, "error": False, "done": None}
        self._worker = self._make_worker(args, lambda ev: self._post(self._on_worker_event, ev),
                                         lambda code, err: self._post(self._on_worker_exit, code, err))
        self._progress.configure(maximum=1, value=0)
        self._stage_lbl.configure(text="단계: 시작 중")
        self._status_lbl.configure(text={"run": "Run 실행 중…", "preview": "Preview 실행 중…",
                                         "export": "Export 실행 중…"}[mode])
        self._set_running(True)
        try:
            self._worker.start()
        except OSError as e:
            self._worker = None
            self._set_running(False)
            messagebox.showerror("Error", f"워커를 시작할 수 없습니다: {e}", parent=self)

    def _set_running(self, running: bool) -> None:
        for key in ("preview", "run", "export"):
            self._btns[key].state(["disabled" if running else "!disabled"])
        self._btns["stop"].state(["!disabled" if running else "disabled"])

    def _post(self, fn: Callable, *args) -> None:
        """읽기 스레드 → Tk 스레드. 탭이 이미 파괴됐으면 무시."""
        try:
            self.after(0, fn, *args)
        except (RuntimeError, tk.TclError):
            pass

    def _on_worker_event(self, ev: dict) -> None:
        try:
            if "stage" in ev:
                total, done = max(int(ev.get("total", 0)), 1), int(ev.get("done", 0))
                self._progress.configure(maximum=total, value=min(done, total))
                name = _STAGE_KO.get(ev["stage"], str(ev["stage"]))
                self._stage_lbl.configure(text=f"단계: {name} ({done}/{ev.get('total', 0)})")
            elif ev.get("event") == "warning":
                self._status_lbl.configure(text=f"경고: {ev.get('message', '')}")
            elif ev.get("event") == "error":
                msg = str(ev.get("message", "알 수 없는 오류"))
                if ev.get("code") == 5 and "requirements-bg.txt" not in msg:
                    msg += f"\n{_DEPS_HINT}"
                self._job["error"] = True
                messagebox.showerror("Error", msg, parent=self)
            elif ev.get("event") == "done":
                self._job["done"] = ev
        except (ValueError, TypeError, tk.TclError):
            pass

    def _on_worker_exit(self, code: int, stderr_last: str) -> None:
        job, self._worker = self._job, None
        self._set_running(False)
        mode, out, done = job.get("mode"), job.get("out"), job.get("done") or {}
        if code == 0:
            self._progress.configure(value=self._progress.cget("maximum"))
            self._stage_lbl.configure(text="단계: 완료")
            if mode == "export":
                msg = format_export_message(done)
                self._status_lbl.configure(text=msg)
                messagebox.showinfo("Export", msg, parent=self)
            elif mode == "preview":
                self.load_results(Path(done.get("preview_dir") or out / "_preview"))
            else:
                self.load_results(out)
            return
        if mode == "run" and (out / "Manifest.jsonl").is_file():
            self.load_results(out)          # 중간까지의 결과도 볼 수 있게
        if job.get("stopping"):
            self._status_lbl.configure(text="중지됨 — 다시 Run하면 이어서 진행합니다")
        elif job.get("error"):
            self._status_lbl.configure(text="실패 (오류 메시지 참고)")
        else:
            msg = f"워커가 종료 코드 {code}로 종료되었습니다." + (f"\n{stderr_last}" if stderr_last else "")
            self._status_lbl.configure(text=msg)
            messagebox.showerror("Error", msg, parent=self)

    def _on_report(self) -> None:
        out = self._vars["output_dir"].get().strip()
        report = load_report(Path(out)) if out else None
        if report is None:
            messagebox.showinfo("Report", "report.json이 없습니다. Run을 완료하면 생성됩니다.", parent=self)
            return
        ReportWindow(self, format_report(report))
