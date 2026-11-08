"""장면 샘플링, 템플릿 프롬프트, LLM 메시지 구성, 프롬프트 정리 (torch 비의존)."""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Protocol

MAX_PROMPT_WORDS = 60
MAX_SANITIZE_PASSES = 10
QUALITY_NEGATIVES = ["blurry", "distorted", "text", "watermark", "extra objects"]

LLM_SYSTEM_PROMPT = (
    "You write background descriptions for an SDXL inpainting model. "
    "Write a single natural English sentence of at most 60 words describing only the indoor "
    "competition venue background. "
    "It must be a fluent sentence, not a list: no labels, no underscores, no colons, "
    "no keyword lists. "
    "Do not mention people, robots, balls, animals, vehicles or any other countable objects. "
    "Never use any forbidden word that the user provides. "
    "Scene elements that describe the camera or photo style (for example \"clean DSLR photo\", "
    "\"smartphone photo with noise\", \"slight motion blur\") describe how the photo looks, "
    "not what is in the scene: never turn them into objects or actions in the scene "
    "(nobody watches a photo or observes through blur). "
    "Put them at the end of the sentence as a short style phrase such as "
    "\"..., shot on a DSLR\". "
    "Mention the keep elements only as context, in the form \"surrounding ...\". "
    "Output the sentence only, with no quotes and no preamble.\n"
    "Example of the desired style: A spacious indoor exhibition hall with a polished concrete "
    "floor, bright overhead LED lighting and colorful sponsor banners hanging along the walls, "
    "captured as a clean DSLR photo."
)


@dataclass
class PromptRequest:
    scene: dict[str, str]
    keep_names: list[str]
    seed: int


class PromptGenerator(Protocol):
    def generate_batch(self, requests: list[PromptRequest]) -> list[str]: ...


def derive_seed(base_seed: int, stem: str, variant: int, attempt: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{stem}:{variant}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def sample_scene(axes: dict[str, list[str]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return {axis: rng.choice(values) for axis, values in axes.items()}


class TemplatePromptGenerator:
    """LLM 없이 장면 값을 이어붙이는 폴백 (req.seed는 사용하지 않는다)."""

    def generate_batch(self, requests: list[PromptRequest]) -> list[str]:
        return [self._one(r) for r in requests]

    @staticmethod
    def _one(req: PromptRequest) -> str:
        parts = [v.strip() for v in req.scene.values() if v and v.strip()]
        if req.keep_names:
            parts.append("surrounding " + " and ".join(req.keep_names))
        parts.append("photorealistic")
        return ", ".join(parts)


def negative_prompt(class_names: list[str]) -> str:
    names = [n.strip() for n in class_names if n and n.strip()]
    return ", ".join(names + QUALITY_NEGATIVES)


_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def _strip_matching_quotes(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    return text


def sanitize_prompt(text: str, class_names: list[str]) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out = _strip_matching_quotes(lines[0]) if lines else ""
    names = sorted({n.strip() for n in class_names if n and n.strip()}, key=len, reverse=True)
    # 단어 사이 공백은 아무 공백 한 칸 이상으로 맞춘다. (f-string 식 안에 백슬래시를 쓰지 않음:
    # Python 3.12 미만 호환)
    ws = r"\s+"
    bodies = [ws.join(re.escape(t) for t in name.split()) for name in names]
    patterns = [
        # \b와 같지만 "c++"처럼 비단어 문자로 끝나는 이름도 다룬다.
        re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)
        for body in bodies
    ]
    # 제거 때문에 새로 붙은 단어가 다른 이름을 만들 수 있으므로 변화가 없을 때까지 반복한다.
    for _ in range(MAX_SANITIZE_PASSES):
        before = out
        for pat in patterns:
            out = pat.sub("", out)
        if out == before:
            break
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    out = re.sub(r",(?:\s*,)+", ",", out)
    out = out.strip(", ")
    words = out.split()
    return " ".join(words[:MAX_PROMPT_WORDS]).rstrip(", ")


def looks_like_keyword_list(text: str, axis_names: list[str]) -> bool:
    """축 이름/밑줄/콜론이 새어 나온 키워드형 출력인지 판별한다 (단순 휴리스틱).

    밑줄/콜론이 있거나, 여러 단어 축 이름("surroundings_floor")이 어디서든 보이거나,
    한 단어 축 이름("lighting")이 쉼표로 구분된 항목 단독으로 나타나면 True.
    한 단어 축 이름은 자연스러운 문장("warm LED lighting")에도 흔히 쓰이므로
    문장 속 단순 등장만으로는 판정하지 않는다.
    """
    if "_" in text or ":" in text:
        return True
    low = text.lower()
    parts = {p.strip(" .") for p in low.split(",")}
    for axis in axis_names:
        name = axis.strip().lower().replace("_", " ")
        if not name:
            continue
        if " " in name:
            if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", low):
                return True
        elif name in parts:
            return True
    return False


def finalize_prompt(raw: str, req: PromptRequest, class_names: list[str],
                    axis_names: list[str]) -> str:
    """LLM 출력을 정리하고, 비었거나 키워드 나열이면 템플릿 프롬프트로 대체한다."""
    out = sanitize_prompt(raw, class_names)
    if not out or looks_like_keyword_list(out, axis_names):
        out = sanitize_prompt(TemplatePromptGenerator._one(req), class_names)
    return out


def build_llm_messages(req: PromptRequest, class_names: list[str]) -> list[dict[str, str]]:
    values = [v.strip() for v in req.scene.values() if v and v.strip()]
    elements = "; ".join(values) if values else "none"
    keep = ", ".join(req.keep_names) if req.keep_names else "none"
    forbidden = ", ".join(class_names) if class_names else "none"
    user = (f"Scene elements: {elements}\nKeep elements: {keep}\n"
            f"Forbidden words: {forbidden}")
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
