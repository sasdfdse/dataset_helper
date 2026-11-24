"""한국어 입력을 영어로 바꾸는 단계 (torch 비의존).

인페인팅(SDXL)과 검출기(Grounding DINO)는 영어만 알아듣는다. 그래서 실행을 시작할 때 한글이 든
입력(유지 항목 이름/검출 프롬프트, 배경 조합 후보 값, 클래스 이름)을 LLM으로 한 번 영어로 바꾸고,
이후 모든 단계는 바뀐 설정을 쓴다. 번역은 {output_dir}/translations.json에 저장해 Preview와 Run,
이어하기가 같은 번역을 쓰게 한다 (원문 → 번역).
"""
from __future__ import annotations

import dataclasses
import json
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from background_swap.common.progress import Progress
from background_swap.common.settings import BackgroundSwapSettings
from background_swap.common.store import atomic_write_text

CACHE_NAME = "translations.json"

_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힣]")
_LABEL_PREFIX = re.compile(r"^(?:english|translation|answer)\s*:\s*", re.IGNORECASE)
_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`"}

TRANSLATE_SYSTEM_PROMPT = (
    "You translate Korean phrases into English. Each phrase describes something in a photo of an "
    "indoor robot or sports competition venue: the venue, lighting, floor, walls, decorations, "
    "camera style, or an object such as equipment, a goal or a ball. "
    "Translate the meaning into a short, common English phrase that an image generation model or "
    "an object detector understands. Never transliterate Korean sounds into Latin letters and never "
    "answer in Chinese or Korean characters. "
    "Output only the English phrase: no quotes, no explanation, no trailing period."
)
# Translation examples (competition-venue context), chosen not to overlap the quality-check words.
TRANSLATE_EXAMPLES = (
    ("의자", "chair"),
    ("파란 커튼", "blue curtain"),
    ("농구 코트", "basketball court"),
    ("어두운 조명", "dim lighting"),
    ("나무 바닥", "wooden floor"),
    ("심판 테이블", "referee table"),
)


class Translator(Protocol):
    def translate_batch(self, texts: list[str]) -> list[str]: ...


class TranslationError(ValueError):
    """번역 결과를 쓸 수 없다 (비었거나 한글이 남음)."""


@dataclass(frozen=True)
class Translation:
    kind: str      # "keep" | "scene" | "class"
    source: str
    text: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "source": self.source, "text": self.text}


def has_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text or ""))


def collect_sources(settings: BackgroundSwapSettings, class_names: list[str]) -> list[tuple[str, str]]:
    """번역할 (종류, 원문) 목록. 한글이 든 것만, 같은 원문은 처음 나온 한 번만."""
    candidates: list[tuple[str, str]] = []
    for item in settings.keep_items:
        candidates += [("keep", item.name), ("keep", item.prompt)]
    for values in settings.scene_axes.values():
        candidates += [("scene", v) for v in values]
    candidates += [("class", n) for n in class_names]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for kind, text in candidates:
        text = (text or "").strip()
        if has_hangul(text) and text not in seen:
            seen.add(text)
            out.append((kind, text))
    return out


def build_translation_messages(text: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": TRANSLATE_SYSTEM_PROMPT}]
    for ko, en in TRANSLATE_EXAMPLES:
        messages += [{"role": "user", "content": ko}, {"role": "assistant", "content": en}]
    return messages + [{"role": "user", "content": text}]


def clean_translation(raw: str) -> str:
    """모델 출력의 첫 줄에서 라벨/따옴표/끝 마침표를 걷어낸다."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    text = _LABEL_PREFIX.sub("", lines[0]) if lines else ""
    for _ in range(3):
        text = text.strip().rstrip(".").strip()
        if len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
            text = text[1:-1]
    return re.sub(r"\s+", " ", text).strip()


def _usable(text) -> bool:
    """영문 번역으로 쓸 수 있는가. 한글뿐 아니라 한자 등 비 ASCII 문자가 섞이면 거부한다."""
    return isinstance(text, str) and bool(text.strip()) and text.isascii() \
        and any(c.isalpha() for c in text)


def _load_cache(path: Path) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and _usable(v)}


def pending_sources(settings: BackgroundSwapSettings, class_names: list[str], cache_path: Path) -> list[str]:
    """아직 번역되지 않은 원문 (비어 있으면 번역 모델을 열 필요가 없다)."""
    cache = _load_cache(cache_path)
    return [text for _, text in collect_sources(settings, class_names) if text not in cache]


def apply_translations(settings: BackgroundSwapSettings, class_names: list[str],
                       mapping: dict[str, str]) -> tuple[BackgroundSwapSettings, list[str]]:
    """원문을 번역으로 바꾼 설정 사본과 클래스 이름. mapping에 없는 값은 그대로 둔다."""
    def tr(text: str) -> str:
        return mapping.get((text or "").strip(), text)

    keep = [dataclasses.replace(k, name=tr(k.name), prompt=tr(k.prompt)) for k in settings.keep_items]
    axes = {axis: [tr(v) for v in values] for axis, values in settings.scene_axes.items()}
    return (dataclasses.replace(settings, keep_items=keep, scene_axes=axes),
            [tr(n) for n in class_names])


def translate_inputs(settings: BackgroundSwapSettings, class_names: list[str],
                     open_translator: Callable[[], AbstractContextManager[Translator]],
                     progress: Progress, cache_path: Path,
                     ) -> tuple[BackgroundSwapSettings, list[str], list[Translation]]:
    """한글 입력을 영어로 바꾼 설정/클래스 이름과 이번에 적용한 번역 목록을 돌려준다.

    저장된 번역이 없는 원문만 모델로 번역한다. 쓸 수 없는 번역이 나오면 그때까지 성공한 번역을
    저장한 뒤 TranslationError를 던진다.
    """
    sources = collect_sources(settings, class_names)
    if not sources:
        return settings, list(class_names), []
    cache = _load_cache(cache_path)
    todo = [text for _, text in sources if text not in cache]
    progress.stage("translate", len(todo))
    if todo:
        try:
            with open_translator() as translator:
                raws = translator.translate_batch(list(todo))
                if len(raws) != len(todo):
                    raise ValueError(f"translate stage: translate_batch returned {len(raws)} results "
                                     f"for {len(todo)} inputs")
                for source, raw in zip(todo, raws):
                    text = clean_translation(raw)
                    if not _usable(text):
                        raise TranslationError(
                            f"'{source}'을(를) 영어로 바꾸지 못했습니다 (모델 출력: '{raw}'). "
                            "이 항목은 영어로 직접 입력하세요.")
                    cache[source] = text
                    progress.advance()
        finally:
            atomic_write_text(cache_path, json.dumps(cache, ensure_ascii=False, indent=2))
    mapping = {text: cache[text] for _, text in sources}
    new_settings, new_names = apply_translations(settings, class_names, mapping)
    return new_settings, new_names, [Translation(kind, text, mapping[text]) for kind, text in sources]
