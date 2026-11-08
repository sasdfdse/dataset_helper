"""Background Swap 결과 탐색/표시용 순수 헬퍼 (Tk 비의존): 목록, 경로, 4패널 이미지, 요약 문구."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from background_swap.composite import diff_image, draw_boxes, mask_overlay
from background_swap.domain import ACCEPTED, EXHAUSTED, REJECTED, SKIPPED, ImageItem
from background_swap.inputs import collect_items, load_image_bgr
from background_swap.store import read_jsonl, read_mask
from core.label import read_labels

_SUMMARY_ORDER = (ACCEPTED, REJECTED, SKIPPED, EXHAUSTED)


@dataclass(frozen=True)
class ResultEntry:
    record: dict
    stem: str
    variant: int
    attempt: int
    status: str
    path: Path


@dataclass
class ResultList:
    entries: list[ResultEntry]
    counts: dict[str, int]

    def summary(self) -> str:
        """상태 라벨용: 'accepted 3 / rejected 1 / skipped 2' (스킵·소진은 목록에 없지만 센다)."""
        parts = [f"{k} {self.counts[k]}" for k in _SUMMARY_ORDER if self.counts.get(k)]
        return " / ".join(parts) if parts else "결과 없음"


@dataclass
class Panels:
    orig: np.ndarray       # 원본 + bbox
    overlay: np.ndarray    # 마스크 오버레이
    result: np.ndarray     # 생성 결과
    diff: np.ndarray       # 차이 (보호 영역 변경은 빨강)


def result_image_path(result_dir: Path, record: dict) -> Path:
    stem, n, k = record["stem"], record["variant"], record.get("attempt")
    if record["status"] == ACCEPTED:
        return Path(result_dir) / "Accepted" / "images" / f"{stem}_bgswap{n}.jpg"
    return Path(result_dir) / "Rejected" / f"{stem}_bgswap{n}_a{k}.jpg"


def build_result_list(result_dir: Path) -> ResultList:
    """Manifest의 accepted/rejected 레코드를 목록화한다. skipped/exhausted는 개수만 센다."""
    result_dir = Path(result_dir)
    records = read_jsonl(result_dir / "Manifest.jsonl")
    counts = Counter(r.get("status") for r in records)
    entries: list[ResultEntry] = []
    for r in records:
        if r.get("status") not in (ACCEPTED, REJECTED) or not isinstance(r.get("stem"), str) \
                or not isinstance(r.get("variant"), int):
            continue
        if r["status"] == REJECTED and not isinstance(r.get("attempt"), int):
            continue
        entries.append(ResultEntry(r, r["stem"], r["variant"], r.get("attempt") or 1,
                                   r["status"], result_image_path(result_dir, r)))
    return ResultList(entries, {k: v for k, v in counts.items() if k})


def index_originals(images_dir: Path, labels_dir: Path) -> dict[str, ImageItem]:
    """stem → 원본/라벨 경로 (frame_stride와 무관하게 전체)."""
    items, _ = collect_items(Path(images_dir), Path(labels_dir), 1)
    return {it.stem: it for it in items}


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def load_panels(result_dir: Path, entry: ResultEntry, item: ImageItem) -> Panels:
    """4패널 BGR 이미지. 원본/결과/마스크를 못 읽으면 FileNotFoundError."""
    orig = load_image_bgr(item.image_path)
    if orig is None:
        raise FileNotFoundError(f"원본 이미지를 읽을 수 없습니다: {item.image_path}")
    result = load_image_bgr(entry.path)
    if result is None:
        raise FileNotFoundError(f"결과 이미지를 읽을 수 없습니다: {entry.path}")
    shape = orig.shape[:2]
    masks_dir = Path(result_dir) / "masks"
    m = {name: _fit(read_mask(masks_dir / f"{entry.stem}_{name}.png"), shape)
         for name in ("object_core", "object_protect", "keep")}
    if result.shape[:2] != shape:
        result_fit = cv2.resize(result, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    else:
        result_fit = result
    protect = m["object_protect"] | m["keep"]
    return Panels(
        orig=draw_boxes(orig, read_labels(item.label_path)),
        overlay=mask_overlay(orig, m["object_core"], m["object_protect"], m["keep"]),
        result=result,
        diff=diff_image(orig, result_fit, protect))


# ── 문구 ──────────────────────────────────────────────────────────

def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return "O" if v else "X"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def _checks_summary(checks) -> str:
    if not isinstance(checks, dict) or not checks:
        return ""
    parts = []
    for key, value in checks.items():
        if key == "vlm":
            if isinstance(value, dict):
                flags = ", ".join(f"{k}={_fmt_value(v)}" for k, v in value.items() if k != "reason")
                reason = value.get("reason")
                parts.append(f"vlm[{flags}{'; ' + str(reason) if reason else ''}]")
            elif value:
                parts.append(f"vlm[{str(value)[:80]}]")
            continue
        parts.append(f"{key}={_fmt_value(value)}")
    return ", ".join(parts)


def describe_record(record: dict) -> str:
    """상태/사유/checks/장면/프롬프트를 여러 줄 문자열로."""
    reasons = record.get("reasons") or []
    scene = record.get("scene") or {}
    lines = [
        f"상태: {record.get('status', '?')}    사유: {', '.join(reasons) if reasons else '–'}",
        f"checks: {_checks_summary(record.get('checks')) or '–'}",
        "장면: " + (", ".join(f"{k}={v}" for k, v in scene.items()) if scene else "–"),
        f"프롬프트: {record.get('prompt') or '–'}",
    ]
    return "\n".join(lines)


def load_report(output_dir: Path) -> Optional[dict]:
    try:
        data = json.loads((Path(output_dir) / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _counts_block(title: str, counts) -> list[str]:
    if not isinstance(counts, dict) or not counts:
        return []
    return [title] + [f"  {k}: {v}" for k, v in counts.items()] + [""]


def format_report(report: dict) -> str:
    """report.json 요약: 상태/사유 집계, 장면 축 분포, 클래스별 recall, 단계별 시간, note."""
    lines: list[str] = []
    lines += _counts_block("[상태]", report.get("status_counts"))
    lines += _counts_block("[거절/스킵 사유]", report.get("reason_counts"))
    scene = report.get("scene_counts") or {}
    for which, title in (("all", "전체"), ("accepted", "accepted")):
        axes = scene.get(which) or {}
        for axis, values in axes.items():
            lines += _counts_block(f"[장면 축 {axis} — {title}]", values)
    recall = report.get("class_recall")
    if isinstance(recall, dict) and recall:
        lines.append("[클래스별 recall (Grounding DINO)]")
        for name, r in recall.items():
            val = r.get("recall") if isinstance(r, dict) else None
            pct = "n/a" if val is None else f"{val * 100:.1f}%"
            hits, total = (r.get("hits"), r.get("total")) if isinstance(r, dict) else (None, None)
            lines.append(f"  {name}: {pct} ({hits}/{total})")
        lines.append("")
    secs = report.get("stage_seconds")
    if isinstance(secs, dict) and secs:
        lines.append("[단계별 소요 시간(초)]")
        lines += [f"  {k}: {v:.1f}" if isinstance(v, (int, float)) else f"  {k}: {v}"
                  for k, v in secs.items()]
        lines.append("")
    if report.get("note"):
        lines.append(f"※ {report['note']}")
    return "\n".join(lines).strip() or "리포트가 비어 있습니다."


def format_export_message(event: dict) -> str:
    if "copied" not in event:
        return "내보내기가 완료되었습니다."
    return (f"내보내기 완료: {event['copied']}쌍 복사, "
            f"이미 있어 건너뜀 {event.get('skipped_existing', 0)}쌍")
