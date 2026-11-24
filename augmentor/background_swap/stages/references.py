"""참고 사진 단계 (torch 비의존): 배경 조합 항목마다 고른 참고 사진을 VLM이 짧은 영어 문구로 설명한다.

LLM(Qwen2.5-3B)은 글만 읽으므로 사진을 직접 넣을 수 없다. 그래서 실행을 시작할 때 VLM(Qwen2.5-VL)이
항목(장소/조명/바닥/주변 풍경/사진 느낌)에 맞춰 사진의 해당 부분만 설명하고, 그 설명을 그 항목의 후보
값에 더한다. 참고 사진은 선택 사항이며, 고르지 않은 항목은 그대로 둔다.
설명은 {output_dir}/references.json에 저장해 Preview/Run/이어하기가 같은 설명을 쓴다.
"""
from __future__ import annotations

import dataclasses
import json
import re
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from background_swap.common.progress import Progress
from background_swap.common.settings import BackgroundSwapSettings, active_reference_images
from background_swap.common.store import atomic_write_text

CACHE_NAME = "references.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_PER_AXIS = 8
MAX_WORDS = 20

_ASK = ("Look at this photo and describe only {what}, as a short English phrase of at most 12 words "
        "that an image generator understands. Do not mention robots, balls, goals, lines or other "
        "objects on the field. Answer with the phrase only.")
AXIS_FOCUS = {
    "venue": "what kind of indoor venue or place this is (for example a school gymnasium or an exhibition hall)",
    "lighting": "the lighting (light sources, color temperature and brightness)",
    "surroundings_floor": "the floor surface around the playing field (material, color, texture)",
    "context": "the surroundings such as walls, decorations, people, furniture or equipment",
    "camera": "the photo style (camera type, sharpness, noise, blur, angle)",
}
_DEFAULT_FOCUS = "the background environment of the place"


def describe_prompt(axis: str) -> str:
    return _ASK.format(what=AXIS_FOCUS.get(axis, _DEFAULT_FOCUS))


class Describer(Protocol):
    def describe(self, image_rgb: np.ndarray, prompt: str) -> str: ...


class ReferencePhotoError(ValueError):
    """참고 사진을 쓸 수 없다 (파일이 없거나 설명을 하나도 만들지 못함)."""


def is_image_file(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def _key(axis: str, path: Path) -> str:
    st = path.stat()
    return f"{axis}|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def clean_description(raw: str) -> str:
    """첫 줄, 공백·따옴표·끝 마침표 정리, 최대 MAX_WORDS 단어. 영문(ASCII)이 아니면 빈 문자열."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    text = " ".join(lines[0].split()) if lines else ""
    text = re.sub(r"^(?:answer|phrase|description)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip().rstrip(".").strip()
    text = " ".join(text.split()[:MAX_WORDS]).rstrip(",;")
    if not text.isascii() or not re.search(r"[A-Za-z]", text):
        return ""
    return text


def _load_cache(path: Path) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v}


def _targets(settings: BackgroundSwapSettings) -> list[tuple[str, Path]]:
    """(항목, 사진) 목록: 체크된 항목만, 항목당 최대 MAX_PER_AXIS장."""
    out = []
    for axis, paths in active_reference_images(settings).items():
        out += [(axis, Path(p)) for p in paths[:MAX_PER_AXIS]]
    return out


def pending_references(settings: BackgroundSwapSettings, cache_path: Path) -> list[tuple[str, Path]]:
    """아직 설명이 없는 (항목, 사진) (비어 있으면 VLM을 열 필요가 없다). 없는 파일은 여기서 세지 않는다."""
    cache = _load_cache(cache_path)
    return [(a, p) for a, p in _targets(settings) if p.is_file() and _key(a, p) not in cache]


def _read_rgb(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def describe_references(settings: BackgroundSwapSettings,
                        open_describer: Callable[[], AbstractContextManager[Describer]],
                        progress: Progress, cache_path: Path,
                        ) -> tuple[BackgroundSwapSettings, list[dict]]:
    """참고 사진 설명을 각 항목 후보에 더한 설정과, 설명 목록 [{axis, file, text}]을 돌려준다."""
    targets = _targets(settings)
    if not targets:
        return settings, []
    missing = [str(p) for _, p in targets if not p.is_file()]
    if missing:
        raise ReferencePhotoError("참고 사진 파일을 찾을 수 없습니다: " + ", ".join(missing))
    cache = _load_cache(cache_path)
    todo = [(a, p) for a, p in targets if _key(a, p) not in cache]
    progress.stage("references", len(todo))
    if todo:
        try:
            with open_describer() as describer:
                for axis, path in todo:
                    rgb = _read_rgb(path)
                    text = clean_description(describer.describe(rgb, describe_prompt(axis))) \
                        if rgb is not None else ""
                    if text:
                        cache[_key(axis, path)] = text
                    progress.advance()
        finally:
            atomic_write_text(cache_path, json.dumps(cache, ensure_ascii=False, indent=2))
    used = [{"axis": a, "file": p.name, "text": cache[_key(a, p)]} for a, p in targets if _key(a, p) in cache]
    failed = sorted({a for a, _ in targets} - {u["axis"] for u in used})
    if failed:
        raise ReferencePhotoError("참고 사진을 설명하지 못한 항목이 있습니다: " + ", ".join(failed)
                                  + ". 사진 파일을 확인하세요.")
    axes = {axis: list(values) for axis, values in settings.scene_axes.items()}
    for u in used:
        if u["text"] not in axes.setdefault(u["axis"], []):
            axes[u["axis"]].append(u["text"])
    return dataclasses.replace(settings, scene_axes=axes), used
