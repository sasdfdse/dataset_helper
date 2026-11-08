"""진행 상황 출력: GUI가 stdout의 JSON Lines를 읽는다."""
from __future__ import annotations

import json
import sys
from typing import Protocol, TextIO


class Progress(Protocol):
    def stage(self, name: str, total: int) -> None: ...
    def advance(self, n: int = 1) -> None: ...
    def event(self, kind: str, **fields) -> None: ...


class JsonLinesProgress:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._name = ""
        self._done = 0
        self._total = 0

    def _emit(self, obj: dict) -> None:
        self._stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._stream.flush()

    def _emit_stage(self) -> None:
        self._emit({"stage": self._name, "done": self._done, "total": self._total})

    def stage(self, name: str, total: int) -> None:
        self._name, self._done, self._total = name, 0, total
        self._emit_stage()

    def advance(self, n: int = 1) -> None:
        self._done += n
        self._emit_stage()

    def event(self, kind: str, **fields) -> None:
        self._emit({"event": kind, **fields})


class NullProgress:
    def stage(self, name: str, total: int) -> None:
        pass

    def advance(self, n: int = 1) -> None:
        pass

    def event(self, kind: str, **fields) -> None:
        pass
