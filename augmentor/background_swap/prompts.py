"""장면 샘플링, 템플릿 프롬프트, LLM 메시지 구성, 프롬프트 정리 (torch 비의존)."""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Protocol

MAX_PROMPT_WORDS = 60
QUALITY_NEGATIVES = ["blurry", "distorted", "text", "watermark", "extra objects"]

_QUOTE_CHARS = "\"'“”‘’"

LLM_SYSTEM_PROMPT = (
    "You write background descriptions for an SDXL inpainting model. "
    "Output exactly one sentence of at most 60 words describing only the indoor "
    "competition venue background. "
    "Do not mention people, robots, balls, animals, vehicles or any other countable objects. "
    "Never use any forbidden word that the user provides. "
    "Mention the keep elements only as context, in the form \"surrounding ...\". "
    "Output the sentence only, with no quotes and no preamble."
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
        parts = ", ".join(req.scene.values())
        if req.keep_names:
            parts += ", surrounding " + " and ".join(req.keep_names)
        return parts + ", photorealistic"


def negative_prompt(class_names: list[str]) -> str:
    return ", ".join(list(class_names) + QUALITY_NEGATIVES)


def sanitize_prompt(text: str, class_names: list[str]) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out = lines[0] if lines else ""
    out = out.strip().strip(_QUOTE_CHARS + " \t").strip()
    for name in class_names:
        name = name.strip()
        if not name:
            continue
        # \b와 같지만 "c++"처럼 비단어 문자로 끝나는 이름도 다룬다.
        out = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    out = re.sub(r",(?:\s*,)+", ",", out)
    out = out.strip(", ")
    words = out.split()
    return " ".join(words[:MAX_PROMPT_WORDS])


def build_llm_messages(req: PromptRequest, class_names: list[str]) -> list[dict[str, str]]:
    scene_lines = "\n".join(f"{axis}: {value}" for axis, value in req.scene.items())
    keep = ", ".join(req.keep_names) if req.keep_names else "none"
    forbidden = ", ".join(class_names) if class_names else "none"
    user = f"{scene_lines}\nKeep elements: {keep}\nForbidden words: {forbidden}"
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
