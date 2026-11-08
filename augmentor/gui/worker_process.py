"""Background Swap CLI 워커를 서브프로세스로 실행하고 JSON Lines 이벤트를 전달한다 (Tk 의존 없음).

콜백(on_event/on_exit)은 읽기 스레드에서 호출된다. Tk 쪽에서는 widget.after(0, ...)로 넘겨야 한다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

AUGMENTOR_DIR = Path(__file__).resolve().parents[1]
KILL_GRACE_SECONDS = 5.0


def parse_event(line: str) -> Optional[dict]:
    """JSON 객체 한 줄이면 dict, 아니면(로그 잡음/배열/빈 줄) None."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


class WorkerProcess:
    def __init__(self, args: list[str], on_event: Callable[[dict], None],
                 on_exit: Callable[[int, str], None], command: Optional[list[str]] = None) -> None:
        self._args = list(args)
        self._on_event = on_event
        self._on_exit = on_exit
        self._command = list(command) if command is not None else [sys.executable, "-m", "background_swap"]
        self._proc: Optional[subprocess.Popen] = None
        self._finished = False
        self._last_stderr = ""

    @property
    def running(self) -> bool:
        """start() 이후 프로세스가 끝날 때까지 True (on_exit 호출 직전에 False가 된다)."""
        return self._proc is not None and not self._finished

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("WorkerProcess can be started only once")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self._proc = subprocess.Popen(
            [*self._command, *self._args], cwd=AUGMENTOR_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._supervise, args=(self._proc,), daemon=True).start()

    def stop(self) -> None:
        """terminate 후 5초 안에 끝나지 않으면 kill. 호출 스레드는 막지 않는다. 실행 중이 아니면 무시."""
        proc = self._proc
        if proc is None or self._finished or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            return

        def _kill_later() -> None:
            try:
                proc.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
        threading.Thread(target=_kill_later, daemon=True).start()

    # ── 읽기 스레드 ───────────────────────────────────────────────

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stderr:
                if line.strip():
                    self._last_stderr = line.strip()
        except (OSError, ValueError):
            pass

    def _supervise(self, proc: subprocess.Popen) -> None:
        err_thread = threading.Thread(target=self._read_stderr, args=(proc,), daemon=True)
        err_thread.start()
        try:
            for line in proc.stdout:
                ev = parse_event(line)
                if ev is not None:
                    try:
                        self._on_event(ev)
                    except Exception:
                        pass            # 콜백 버그가 읽기 루프를 죽이지 않게 한다
        except (OSError, ValueError):
            pass
        code = proc.wait()
        err_thread.join(timeout=5)
        self._finished = True
        try:
            self._on_exit(code, self._last_stderr)
        except Exception:
            pass
