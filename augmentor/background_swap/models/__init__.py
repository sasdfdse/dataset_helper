"""실제 Hugging Face 모델 래퍼 패키지 (Background Swap에서 torch를 import하는 유일한 곳).

구현은 `background_swap.models.hf`에 있다. 공개 이름은 여기서 다시 내보내되, 지연(lazy) 방식이다:
이 `__init__`은 pytest가 패키지를 수집할 때(시스템 python3, torch 없음)에도 import되므로
최상단에서 torch/transformers/diffusers를 끌어오면 안 된다. 이름에 처음 접근할 때 hf를 import한다.

주의: 팩토리가 사용하는 클래스를 monkeypatch하려면 이 패키지가 아니라
`background_swap.models.hf`의 이름을 대상으로 해야 한다.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 정적 분석용; 런타임에는 import하지 않는다
    from background_swap.models.hf import (HFInpainter, HFModelFactory, HFPromptGenerator,
                                           HFSegmenter, HFTextDetector, HFValidator,
                                           match_phrase, oom_retry, preflight, release)

__all__ = [
    "HFInpainter", "HFModelFactory", "HFPromptGenerator", "HFSegmenter", "HFTextDetector",
    "HFValidator", "match_phrase", "oom_retry", "preflight", "release",
]


def __getattr__(name: str):
    if name in __all__:
        return getattr(importlib.import_module("background_swap.models.hf"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
