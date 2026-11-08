"""Background Swap 탭 실행/결과 흐름 (가짜 WorkerProcess, 모델 없음). DISPLAY 필요."""
import os
import threading
import time
from pathlib import Path

import pytest

from background_swap.settings import BackgroundSwapSettings, load_settings, save_settings
from tests.test_bg_results import make_output

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no DISPLAY")


class FakeWorker:
    """생성 시 넘어온 args를 기록하고, start()에서 스크립트된 이벤트를 별도 스레드로 재생한다."""
    script: list = []
    exit_code = 0
    stderr_last = ""
    hold = None            # threading.Event: 설정돼 있으면 set될 때까지 종료하지 않음
    instances: list = []

    def __init__(self, args, on_event, on_exit):
        self.args, self._on_event, self._on_exit = args, on_event, on_exit
        self.stopped = False
        self.running = False
        FakeWorker.instances.append(self)

    def start(self):
        self.running = True
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        for ev in FakeWorker.script:
            self._on_event(ev)
            time.sleep(0.01)
        if FakeWorker.hold is not None:
            FakeWorker.hold.wait(5)
        self.running = False
        self._on_exit(FakeWorker.exit_code, FakeWorker.stderr_last)

    def stop(self):
        self.stopped = True
        if FakeWorker.hold is not None:
            FakeWorker.exit_code = -15
            FakeWorker.hold.set()


@pytest.fixture
def env(tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import filedialog, messagebox
    monkeypatch.setattr("gui.background_state.UI_STATE_PATH", tmp_path / "ui_state.json")
    FakeWorker.script, FakeWorker.exit_code, FakeWorker.stderr_last = [], 0, ""
    FakeWorker.hold, FakeWorker.instances = None, []
    shown = {"error": [], "info": [], "yesno": []}
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: shown["error"].append(a[1]))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: shown["info"].append(a[1]))
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: (shown["yesno"].append(a), True)[1])
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: "")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open display")
    root.withdraw()
    from gui.background_tab import BackgroundTab
    monkeypatch.setattr("gui.background_tab.WorkerProcess", FakeWorker)
    images, labels, out = make_output(tmp_path)
    tab = BackgroundTab(root)
    tab.pack()
    tab.apply_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out))
    tab.update()
    yield tab, shown, (images, labels, out)
    root.destroy()


def pump(tab, cond, timeout=5.0):
    """실제 앱처럼 mainloop를 돌린다 (다른 스레드의 after()는 mainloop 안에서만 허용된다)."""
    root = tab.winfo_toplevel()
    deadline = time.time() + timeout

    def check():
        if cond() or time.time() > deadline:
            root.quit()
        else:
            root.after(10, check)
    root.after(10, check)
    root.mainloop()
    return bool(cond())


def btn_disabled(tab, key):
    return tab._btns[key].instate(["disabled"])


@needs_display
def test_run_replays_events_updates_progress_and_loads_results(env):
    tab, shown, (images, labels, out) = env
    FakeWorker.script = [{"stage": "masks", "done": 1, "total": 2},
                         {"event": "warning", "message": "경고 문구"},
                         {"stage": "inpaint", "done": 2, "total": 4},
                         {"event": "done"}]
    tab._btns["run"].invoke()
    assert tab.is_running() and btn_disabled(tab, "run") and btn_disabled(tab, "preview") \
        and btn_disabled(tab, "export") and not btn_disabled(tab, "stop")
    assert pump(tab, lambda: not tab.is_running())
    w = FakeWorker.instances[0]
    assert w.args[0] == "run" and w.args[1] == "--settings"
    saved = load_settings(Path(w.args[2]))
    assert Path(w.args[2]) == out / "run_settings.json" and saved.output_dir == out
    assert not saved.allow_non_train
    assert "완료" in tab._stage_lbl.cget("text")
    assert float(tab._progress.cget("value")) == float(tab._progress.cget("maximum"))
    assert not btn_disabled(tab, "run") and btn_disabled(tab, "stop")
    # 결과 4패널 로드
    assert tab._results_mode and tab._nav_lbl.cget("text") == "1 / 2"
    for lbl in (tab._orig_lbl, tab._mask_lbl, tab._result_lbl, tab._diff_lbl):
        assert lbl.image is not None
    assert "accepted 1" in tab._status_lbl.cget("text") and "skipped 1" in tab._status_lbl.cget("text")
    assert "gym" in tab._detail_lbl.cget("text") and "protected_changed" not in tab._detail_lbl.cget("text")
    tab._step(1)
    assert tab._nav_lbl.cget("text") == "2 / 2"
    assert "protected_changed" in tab._detail_lbl.cget("text")
    assert not shown["error"]


@needs_display
def test_stage_events_drive_progress_and_label(env):
    tab, _, _ = env
    FakeWorker.hold = threading.Event()
    FakeWorker.script = [{"stage": "masks", "done": 1, "total": 2}, {"stage": "inpaint", "done": 3, "total": 4}]
    tab._btns["run"].invoke()
    assert pump(tab, lambda: "인페인팅" in tab._stage_lbl.cget("text"))
    assert "(3/4)" in tab._stage_lbl.cget("text")
    assert float(tab._progress.cget("maximum")) == 4 and float(tab._progress.cget("value")) == 3
    FakeWorker.hold.set()
    assert pump(tab, lambda: not tab.is_running())


@needs_display
def test_stop_button_stops_worker_and_restores_buttons(env):
    tab, shown, _ = env
    FakeWorker.hold = threading.Event()
    tab._btns["run"].invoke()
    tab.update()
    tab._btns["stop"].invoke()
    assert FakeWorker.instances[0].stopped
    assert pump(tab, lambda: not tab.is_running())
    assert "중지됨" in tab._status_lbl.cget("text")
    assert not shown["error"] and not btn_disabled(tab, "run")


@needs_display
def test_error_event_and_deps_hint(env):
    tab, shown, _ = env
    FakeWorker.script = [{"event": "error", "code": 5, "message": "필수 패키지를 불러올 수 없습니다"}]
    FakeWorker.exit_code = 5
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert len(shown["error"]) == 1                 # 종료 코드로 중복 표시하지 않음
    assert "필수 패키지" in shown["error"][0] and "pip install -r requirements-bg.txt" in shown["error"][0]
    # 메시지에 이미 안내가 있으면 덧붙이지 않는다
    shown["error"].clear()
    FakeWorker.script = [{"event": "error", "code": 5, "message": "x: pip install -r requirements-bg.txt"}]
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert shown["error"][0].count("requirements-bg.txt") == 1


@needs_display
def test_nonzero_exit_without_error_event_shows_stderr(env):
    tab, shown, _ = env
    FakeWorker.exit_code, FakeWorker.stderr_last = 1, "Traceback: boom"
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert len(shown["error"]) == 1 and "Traceback: boom" in shown["error"][0] \
        and "종료 코드 1" in shown["error"][0] and not btn_disabled(tab, "run")


@needs_display
def test_invalid_settings_block_launch(env):
    tab, shown, _ = env
    tab._vars["output_dir"].set("")
    tab._btns["run"].invoke()
    assert shown["error"] and not FakeWorker.instances and not tab.is_running()


@needs_display
def test_non_train_confirmation_sets_allow_for_that_run(env, tmp_path, monkeypatch):
    from tkinter import messagebox
    tab, shown, (images, labels, out) = env
    val = tmp_path / "val" / "images"
    val.parent.mkdir()
    images.rename(val)
    tab._vars["images_dir"].set(str(val))
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert len(shown["yesno"]) == 1
    assert load_settings(out / "run_settings.json").allow_non_train is True
    assert tab.build_settings().allow_non_train is False       # 영속 옵션이 아님
    # 거절하면 실행하지 않는다
    FakeWorker.instances.clear()
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: False)
    tab._btns["run"].invoke()
    assert not FakeWorker.instances


@needs_display
def test_mismatch_overwrite_resets_output_then_runs(env, monkeypatch):
    tab, shown, (images, labels, out) = env
    save_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out, seed=99),
                  out / "run_settings.json")
    monkeypatch.setattr(tab, "_ask_mismatch", lambda diff: ("overwrite", asked.append(diff))[0])
    asked = []
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert asked == [["seed"]]
    assert not (out / "Manifest.jsonl").exists() and not (out / "Accepted").exists() \
        and not (out / "masks").exists()                       # 실제 reset_output_dir 수행 (가짜 워커는 새로 쓰지 않음)
    assert load_settings(out / "run_settings.json").seed == 0  # 새 설정이 저장됨
    assert FakeWorker.instances


@needs_display
def test_mismatch_cancel_and_new_folder(env, tmp_path, monkeypatch):
    from tkinter import filedialog
    tab, shown, (images, labels, out) = env
    save_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out, seed=99),
                  out / "run_settings.json")
    monkeypatch.setattr(tab, "_ask_mismatch", lambda diff: "cancel")
    tab._btns["run"].invoke()
    assert not FakeWorker.instances and (out / "Manifest.jsonl").exists()   # 아무것도 지우지 않음
    new = tmp_path / "fresh"
    monkeypatch.setattr(tab, "_ask_mismatch", lambda diff: "new")
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: str(new))
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert (out / "Manifest.jsonl").exists()                 # 기존 출력 보존
    assert FakeWorker.instances[0].args[2] == str(new / "run_settings.json")
    assert tab._vars["output_dir"].get() == str(new)


@needs_display
def test_mismatch_calls_reset_output_dir(env, monkeypatch):
    tab, shown, (images, labels, out) = env
    save_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out, seed=99),
                  out / "run_settings.json")
    calls = []
    monkeypatch.setattr("gui.background_tab.reset_output_dir", lambda p: calls.append(Path(p)))
    monkeypatch.setattr(tab, "_ask_mismatch", lambda diff: "overwrite")
    tab._btns["run"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert calls == [out]


@needs_display
def test_preview_uses_separate_settings_file_and_loads_preview_dir(env, tmp_path):
    tab, shown, (images, labels, out) = env
    prev_dir = tmp_path / "prevout"
    (prev_dir / "masks").mkdir(parents=True)
    import shutil
    for f in (out / "masks").iterdir():
        shutil.copy(f, prev_dir / "masks" / f.name)
    shutil.copytree(out / "Accepted", prev_dir / "Accepted")
    (prev_dir / "Manifest.jsonl").write_text(
        (out / "Manifest.jsonl").read_text().splitlines()[0] + "\n")
    FakeWorker.script = [{"event": "done", "preview_dir": str(prev_dir)}]
    tab._btns["preview"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    args = FakeWorker.instances[0].args
    assert args[0] == "preview" and args[3:] == ["--n", "3"]
    assert not (out / "run_settings.json").exists()        # 이어하기 기준 파일을 건드리지 않음
    assert tab._results_mode and tab._result_dir == prev_dir and tab._nav_lbl.cget("text") == "1 / 1"


@needs_display
def test_export_flow_and_done_message(env, monkeypatch, tmp_path):
    from tkinter import filedialog
    tab, shown, (images, labels, out) = env
    tab._btns["export"].invoke()                     # run_settings.json 없음
    assert shown["error"] and not FakeWorker.instances
    save_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out),
                  out / "run_settings.json")
    dirs = iter([str(tmp_path / "train_images"), str(tmp_path / "train_labels")])
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: next(dirs))
    FakeWorker.script = [{"event": "done", "copied": 4, "skipped_existing": 1}]
    tab._btns["export"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    w = FakeWorker.instances[0]
    assert w.args == ["export", "--settings", str(out / "run_settings.json"),
                      "--images-dir", str(tmp_path / "train_images"),
                      "--labels-dir", str(tmp_path / "train_labels")]
    assert shown["info"] and "4" in shown["info"][0] and "1" in shown["info"][0]
    assert not shown["yesno"]


@needs_display
def test_export_non_train_target_confirmed_writes_allow_settings(env, monkeypatch, tmp_path):
    from tkinter import filedialog
    tab, shown, (images, labels, out) = env
    save_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out),
                  out / "run_settings.json")
    dirs = iter([str(tmp_path / "val" / "images"), str(tmp_path / "val" / "labels")])
    monkeypatch.setattr(filedialog, "askdirectory", lambda **k: next(dirs))
    tab._btns["export"].invoke()
    assert pump(tab, lambda: not tab.is_running())
    assert len(shown["yesno"]) == 1
    sp = Path(FakeWorker.instances[0].args[2])
    assert sp.name == "_export_settings.json" and load_settings(sp).allow_non_train is True
    assert load_settings(out / "run_settings.json").allow_non_train is False


@needs_display
def test_report_button(env, monkeypatch):
    import json
    tab, shown, (images, labels, out) = env
    tab._btns["report"].invoke()                     # report.json 없음
    assert shown["info"]
    (out / "report.json").write_text(json.dumps(
        {"status_counts": {"accepted": 1}, "class_recall": {"ball": {"hits": 1, "total": 1, "recall": 1.0}},
         "note": "GDINO 미검출은 새 객체가 없다는 뜻이 아닙니다."}))
    made = []
    import gui.background_run as br
    monkeypatch.setattr(br, "ReportWindow", lambda parent, text: made.append(text))
    tab._btns["report"].invoke()
    assert made and "ball" in made[0] and "100.0%" in made[0] and "GDINO" in made[0]


@needs_display
def test_output_dir_with_manifest_loads_results_and_back_button(env):
    tab, _, (images, labels, out) = env
    assert not tab._results_mode
    tab._on_output_changed()
    assert tab._results_mode and tab._nav_lbl.cget("text") == "1 / 2"
    tab._step(-1)
    assert tab._nav_lbl.cget("text") == "2 / 2"
    tab._back_btn.invoke()
    assert not tab._results_mode and tab._result_lbl.image is None
    tab.update()


@needs_display
def test_destroy_stops_running_worker(env):
    tab, _, _ = env
    FakeWorker.hold = threading.Event()
    tab._btns["run"].invoke()
    tab.update()
    tab.destroy()
    assert FakeWorker.instances[0].stopped
    FakeWorker.hold.set()


@needs_display
def test_reset_failure_shows_error_and_aborts(env, monkeypatch):
    tab, shown, (images, labels, out) = env
    save_settings(BackgroundSwapSettings(images_dir=images, labels_dir=labels, output_dir=out, seed=99),
                  out / "run_settings.json")

    def boom(p):
        raise PermissionError("denied")
    monkeypatch.setattr("gui.background_tab.reset_output_dir", boom)
    monkeypatch.setattr(tab, "_ask_mismatch", lambda diff: "overwrite")
    tab._btns["run"].invoke()
    assert len(shown["error"]) == 1 and "denied" in shown["error"][0]
    assert not FakeWorker.instances and not tab.is_running()


@needs_display
def test_diff_panel_title_notes_jpeg_fallback(env):
    tab, _, (images, labels, out) = env
    (out / "generated" / "a_bgswap1_a1.png").unlink()
    tab.load_results(out)
    assert "JPEG" in tab._titles[3][0].cget("text")
    tab._step(1)                                  # PNG가 있는 레코드
    assert "JPEG" not in tab._titles[3][0].cget("text")
