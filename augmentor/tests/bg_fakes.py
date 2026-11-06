"""Background Swap 테스트 공용 가짜 모델/이미지 (torch 비의존)."""
from __future__ import annotations

import numpy as np

from background_swap.domain import Detection

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
