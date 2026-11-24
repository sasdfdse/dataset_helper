"""앱 전체 설정: 기본 폴더, 마지막으로 연 폴더, 탭별 마지막 경로 (앱을 꺼도 남는다).

- 기본 폴더: 모든 Browse 창이 여기서 열린다 (칸에 이미 경로가 있으면 그 경로 쪽이 먼저).
- 탭별 마지막 경로: Augmentation/Video Tool 탭의 경로 칸을 다음 실행 때 다시 채운다
  (Background Swap 탭은 자체 UI 상태 파일에 모든 설정을 저장한다).
- 저장 위치: ~/.config/dataset_helper/app_prefs.json (UI 상태 파일과 같은 폴더)

파일 대화상자는 이 모듈의 ask_* 함수로 연다. 시작 폴더를 정하고, 고른 폴더를 기억한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tkinter import filedialog

from background_swap.common.store import atomic_write_text
from gui import background_state

PREFS_NAME = "app_prefs.json"


def prefs_path() -> Path:
    # read at call time (tests point UI_STATE_PATH at a temp dir)
    return Path(background_state.UI_STATE_PATH).parent / PREFS_NAME


def _load() -> dict:
    try:
        data = json.loads(prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    path = prefs_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
    except OSError:
        pass


def default_root() -> Optional[Path]:
    value = _load().get("default_root")
    return Path(value) if isinstance(value, str) and value and Path(value).is_dir() else None


def set_default_root(path: Optional[Path]) -> None:
    data = _load()
    data["default_root"] = str(path) if path else None
    _save(data)


def recall_path(key: str) -> str:
    paths = _load().get("paths")
    value = paths.get(key) if isinstance(paths, dict) else None
    return value if isinstance(value, str) else ""


def remember_path(key: str, value: str) -> None:
    data = _load()
    paths = data.get("paths") if isinstance(data.get("paths"), dict) else {}
    paths[key] = value or ""
    data["paths"] = paths
    _save(data)


def _existing_dir(value) -> Optional[Path]:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    for cand in (p, p.parent):
        if cand.is_dir() and str(cand) not in ("", "."):
            return cand
    return None


def start_dir(current: str = "") -> str:
    """Browse 창의 시작 폴더: 칸의 현재 경로 → 기본 폴더 → 마지막으로 연 폴더 → 홈."""
    for cand in (_existing_dir(current), default_root(), _existing_dir(_load().get("last_dir"))):
        if cand is not None:
            return str(cand)
    return str(Path.home())


def _remember_dir(path: str) -> None:
    folder = _existing_dir(path)
    if folder is not None:
        data = _load()
        data["last_dir"] = str(folder)
        _save(data)


def ask_directory(title: str, current: str = "", **kw) -> str:
    path = filedialog.askdirectory(title=title, initialdir=start_dir(current), **kw)
    if path:
        _remember_dir(path)
    return path or ""


def ask_open_file(title: str, current: str = "", **kw) -> str:
    path = filedialog.askopenfilename(title=title, initialdir=start_dir(current), **kw)
    if path:
        _remember_dir(path)
    return path or ""


def ask_open_files(title: str, current: str = "", **kw) -> list[str]:
    paths = filedialog.askopenfilenames(title=title, initialdir=start_dir(current), **kw)
    if paths:
        _remember_dir(paths[0])
    return list(paths or [])


def ask_save_file(title: str, current: str = "", **kw) -> str:
    path = filedialog.asksaveasfilename(title=title, initialdir=start_dir(current), **kw)
    if path:
        _remember_dir(path)
    return path or ""
