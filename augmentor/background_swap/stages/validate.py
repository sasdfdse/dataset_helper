"""생성 이미지 검증 로직 (torch 의존 없음).

- rule_checks: 결정적 픽셀 규칙 (크기, 보호 영역 불변, 최소 변화량)
- find_new_objects: 검출기 출력에서 새로 생긴 객체 필터링
- parse_vlm_json / combine: VLM 판정 파싱과 최종 판정 결합
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np

from background_swap.common.domain import Detection, ValidationResult

VLM_PROMPT = (
    "You are checking a synthetic image whose background was replaced by an image generator. "
    "The image has bounding boxes drawn on it marking the original target objects.\n"
    "Target object classes: {class_names}.\n"
    "Answer these questions about the image:\n"
    "1. natural_indoor_venue: does the background look like a natural, realistic indoor venue "
    "(true/false)?\n"
    "2. boundary_artifacts: are there visible artifacts, seams, halos, or distortions around the "
    "boundaries of the marked boxes (true/false)?\n"
    "3. new_objects: is there any object of the same kind as the target classes visible OUTSIDE "
    "the marked boxes (true/false)?\n"
    "Output exactly one JSON object and nothing else, in this format:\n"
    '{{"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": false, '
    '"reason": "short explanation"}}'
)

_VLM_BOOL_KEYS = ("natural_indoor_venue", "boundary_artifacts", "new_objects")


class ImageValidator(Protocol):
    def judge_batch(self, images_rgb: list[np.ndarray], prompt: str) -> list[str]: ...


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """xyxy 박스 IoU. 합집합 면적이 0이면 0.0."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def rule_checks(
    orig: np.ndarray,
    out: np.ndarray,
    protect: np.ndarray,
    inpaint: np.ndarray,
    min_change: float,
) -> tuple[list[str], dict[str, object]]:
    if orig.shape != out.shape:
        return ["size_mismatch"], {"protected_identical": False, "mean_change": 0.0}

    diff = np.abs(orig.astype(np.int16) - out.astype(np.int16))
    protected_identical = not bool(diff[protect].any())
    if inpaint.any():
        mean_change = float(diff[inpaint].mean(axis=-1).mean())
    else:
        mean_change = 0.0

    reasons: list[str] = []
    if not protected_identical:
        reasons.append("protected_changed")
    if mean_change < min_change:
        reasons.append("too_little_change")
    return reasons, {"protected_identical": protected_identical, "mean_change": mean_change}


def find_new_objects(
    dets: list[Detection],
    label_boxes: list[tuple[int, int, int, int]],
    inpaint: np.ndarray,
    score_thr: float = 0.35,
    iou_thr: float = 0.3,
    inpaint_frac: float = 0.5,
) -> list[Detection]:
    h, w = inpaint.shape[:2]
    found: list[Detection] = []
    for d in dets:
        if d.score < score_thr:
            continue
        x1 = max(0, int(round(d.box[0])))
        y1 = max(0, int(round(d.box[1])))
        x2 = min(w, int(round(d.box[2])))
        y2 = min(h, int(round(d.box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        if any(iou(d.box, lb) >= iou_thr for lb in label_boxes):
            continue
        frac = float(inpaint[y1:y2, x1:x2].mean())
        if frac >= inpaint_frac:
            found.append(d)
    return found


def _extract_json_candidates(text: str):
    """각 `{`에서 시작해 괄호 깊이로 대응하는 `}`까지의 부분 문자열을 순서대로 yield."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            yield text[start:end + 1]
        start = text.find("{", start + 1)


def parse_vlm_json(text: str) -> dict | None:
    for cand in _extract_json_candidates(text):
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and all(isinstance(obj.get(k), bool) for k in _VLM_BOOL_KEYS):
            return obj
    return None


def combine(
    rule_reasons: list[str],
    checks: dict,
    gdino_new: list[Detection] | None,
    vlm_text: str | None,
) -> ValidationResult:
    reasons = list(rule_reasons)
    out_checks: dict[str, object] = dict(checks)
    out_checks["gdino_new_count"] = None if gdino_new is None else len(gdino_new)
    out_checks["vlm"] = None

    if gdino_new:
        reasons.append("new_object:gdino")

    if vlm_text is not None:
        parsed = parse_vlm_json(vlm_text)
        if parsed is None:
            reasons.append("vlm_unparsable")
            out_checks["vlm"] = vlm_text
        else:
            out_checks["vlm"] = parsed
            if not parsed["natural_indoor_venue"]:
                reasons.append("vlm_unnatural")
            if parsed["boundary_artifacts"]:
                reasons.append("vlm_boundary_artifacts")
            if parsed["new_objects"]:
                reasons.append("new_object:vlm")

    return ValidationResult(accepted=not reasons, reasons=reasons, checks=out_checks)
