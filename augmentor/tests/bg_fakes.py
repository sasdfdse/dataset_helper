"""Background Swap 테스트 공용 가짜 모델/이미지 (torch 비의존)."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from background_swap.domain import Detection
from background_swap.prompts import TemplatePromptGenerator
from background_swap.settings import BackgroundSwapSettings

GREEN_BGR = (40, 160, 40)
GREEN_HSV_RANGE = ([35, 40, 40], [85, 255, 255])


class FakeSegmenter:
    """박스마다 박스 사각형을 채운 bool 마스크를 반환하고 호출을 기록한다."""

    def __init__(self):
        self.calls: list[list[tuple[float, float, float, float]]] = []

    def segment_boxes(self, image_rgb, boxes):
        self.calls.append(list(boxes))
        h, w = image_rgb.shape[:2]
        out = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros((h, w), bool)
            m[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
            out.append(m)
        return out


class FakeDetector:
    """phrases 각각의 검출 목록을 이어 붙여 반환한다."""

    def __init__(self, dets_by_phrase: dict[str, list[Detection]]):
        self.dets_by_phrase = dets_by_phrase
        self.calls: list[tuple[list[str], float, float]] = []

    def detect(self, image_rgb, phrases, box_threshold, text_threshold):
        self.calls.append((list(phrases), box_threshold, text_threshold))
        out: list[Detection] = []
        for p in phrases:
            out.extend(self.dets_by_phrase.get(p, []))
        return out


def field_image(h: int = 100, w: int = 100) -> np.ndarray:
    """회색(128) 배경, 위쪽 30% 아래는 GREEN_BGR."""
    img = np.full((h, w, 3), 128, np.uint8)
    img[int(h * 0.3):, :] = GREEN_BGR
    return img


class FakeInpainter:
    """입력을 반전(255 - img)해 반환. fail_after번 성공 후 RuntimeError, raise_exc면 첫 호출에 그 예외."""

    def __init__(self, native_px: int = 64, fail_after: int | None = None,
                 raise_exc: Exception | None = None):
        self.native_px = native_px
        self.fail_after = fail_after
        self.raise_exc = raise_exc
        self.calls = 0

    def inpaint(self, image_rgb, mask, prompt, negative_prompt, seed):
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("FakeInpainter: simulated crash")
        self.calls += 1
        return 255 - image_rgb


class FakeValidator:
    """judge_batch가 이미지마다 다음 응답을 반환 (소진 시 마지막 응답 반복)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._i = 0
        self.calls: list[tuple[int, str]] = []   # (이미지 수, 프롬프트)

    def judge_batch(self, images_rgb, prompt):
        self.calls.append((len(images_rgb), prompt))
        out = []
        for _ in images_rgb:
            out.append(self.responses[min(self._i, len(self.responses) - 1)])
            self._i += 1
        return out


class FakeFactory:
    """ModelFactory 가짜. 컨텍스트 진입 시 메서드 이름을 opened에 기록한다."""

    def __init__(self, segmenter, detector, prompt_gen, inpainter, validator):
        self.segmenter = segmenter
        self.detector_obj = detector
        self.prompt_gen = prompt_gen
        self.inpainter_obj = inpainter
        self.validator_obj = validator
        self.opened: list[str] = []

    @classmethod
    def default(cls, **overrides) -> "FakeFactory":
        kwargs = dict(segmenter=FakeSegmenter(), detector=FakeDetector({}),
                      prompt_gen=TemplatePromptGenerator(), inpainter=FakeInpainter(),
                      validator=FakeValidator([vlm_json(True)]))
        unknown = set(overrides) - set(kwargs)
        if unknown:
            raise TypeError(f"unknown FakeFactory overrides: {sorted(unknown)}")
        kwargs.update(overrides)
        return cls(**kwargs)

    @contextmanager
    def mask_models(self):
        self.opened.append("mask_models")
        yield self.segmenter, self.detector_obj

    @contextmanager
    def prompt_model(self):
        self.opened.append("prompt_model")
        yield self.prompt_gen

    @contextmanager
    def inpainter(self):
        self.opened.append("inpainter")
        yield self.inpainter_obj

    @contextmanager
    def detector(self):
        self.opened.append("detector")
        yield self.detector_obj

    @contextmanager
    def validator(self):
        self.opened.append("validator")
        yield self.validator_obj


def vlm_json(ok: bool) -> str:
    return json.dumps({"natural_indoor_venue": bool(ok), "boundary_artifacts": False,
                       "new_objects": False, "reason": "fake"})


def make_settings(tmp_path, n_images: int = 1, size: tuple[int, int] = (64, 48),
                  rgba: bool = False, **overrides) -> BackgroundSwapSettings:
    """tmp_path/train/{images,labels}에 노이즈 이미지 img_000…와 라벨을 만든다. size = (w, h)."""
    root = Path(tmp_path) / "train"
    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    rng = np.random.default_rng(0)
    for i in range(n_images):
        stem = f"img_{i:03d}"
        if rgba:
            img = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
            img[..., 3] = 255
            cv2.imwrite(str(images_dir / f"{stem}.png"), img)
        else:
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), img)
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return BackgroundSwapSettings(images_dir=images_dir, labels_dir=labels_dir,
                                  output_dir=Path(tmp_path) / "out", **overrides)
