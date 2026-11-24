"""Background Swap 결과 탐색/표시용 순수 헬퍼 (Tk 비의존): 목록, 경로, 4패널 이미지, 요약 문구."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from background_swap.stages.composite import diff_image, draw_boxes, mask_overlay
from background_swap.common.settings import axis_label
from background_swap.common.domain import ACCEPTED, EXHAUSTED, REJECTED, SKIPPED, ImageItem
from background_swap.dataset.inputs import collect_items, load_image_bgr
from background_swap.common.store import read_jsonl, read_mask
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
    orig: np.ndarray
    overlay: np.ndarray
    result: np.ndarray
    diff: np.ndarray       # diff (changes in protected areas shown in red)
    diff_note: str = ""    # if set, appended to the diff panel title as a warning


DIFF_JPEG_NOTE = "(JPEG 기준 — 압축 오차 포함)"


def generated_png_path(result_dir: Path, entry: "ResultEntry") -> Path:
    """무손실 생성 결과 (accepted/rejected 공통, 시도 번호 포함)."""
    return Path(result_dir) / "generated" / f"{entry.stem}_bgswap{entry.variant}_a{entry.attempt}.png"


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


def originals_from_items(result_dir: Path) -> dict[str, ImageItem] | None:
    """result_dir/items.jsonl(실행 당시 입력 목록)의 stem → 원본/라벨 경로. 파일이 없으면 None.

    결과를 만든 입력 폴더가 지금 GUI에 지정된 폴더와 달라도 (같은 stem이 겹쳐도) 올바른 원본을 쓴다.
    """
    path = Path(result_dir) / "items.jsonl"
    if not path.is_file():
        return None
    out: dict[str, ImageItem] = {}
    for r in read_jsonl(path):
        stem, img, lbl = r.get("stem"), r.get("image_path"), r.get("label_path")
        if not (isinstance(stem, str) and isinstance(img, str) and isinstance(lbl, str)):
            continue
        w, h = r.get("width"), r.get("height")
        out.setdefault(stem, ImageItem(stem, Path(img), Path(lbl),
                                       w if isinstance(w, int) else 0,
                                       h if isinstance(h, int) else 0))
    return out


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
    # Diff against the lossless PNG (JPEG artifacts would show up red in protected areas).
    # Fall back to the JPEG and say so if the PNG is missing.
    lossless = load_image_bgr(generated_png_path(result_dir, entry))
    note = ""
    if lossless is None:
        lossless, note = result, DIFF_JPEG_NOTE
    if lossless.shape[:2] != shape:
        result_fit = cv2.resize(lossless, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    else:
        result_fit = lossless
    protect = m["object_protect"] | m["keep"]
    return Panels(
        orig=draw_boxes(orig, read_labels(item.label_path)),
        overlay=mask_overlay(orig, m["object_core"], m["object_protect"], m["keep"]),
        result=result,
        diff=diff_image(orig, result_fit, protect), diff_note=note)


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
        "배경 조합: " + (", ".join(f"{axis_label(k)}={v}" for k, v in scene.items()) if scene else "–"),
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
    """report.json 요약: 상태/사유 집계, 배경 조합 분포, 클래스별 recall, 단계별 시간, note."""
    lines: list[str] = []
    lines += _counts_block("[상태]", report.get("status_counts"))
    lines += _counts_block("[거절/스킵 사유]", report.get("reason_counts"))
    scene = report.get("scene_counts") or {}
    for which, title in (("all", "전체"), ("accepted", "accepted")):
        axes = scene.get(which) or {}
        for axis, values in axes.items():
            lines += _counts_block(f"[배경 조합: {axis_label(axis)} — {title}]", values)
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
    avg = report.get("avg_seconds_per_generation")
    if isinstance(avg, (int, float)) and not isinstance(avg, bool):
        lines += ["[이미지당 평균 생성 시간]", f"  {avg:.2f}초 (인페인팅 시간 / 생성 시도 수)", ""]
    if report.get("note"):
        lines.append(f"※ {report['note']}")
    return "\n".join(lines).strip() or "리포트가 비어 있습니다."


_TRANSLATION_GROUPS = (("keep", "유지 항목"), ("scene", "배경 조합"), ("class", "클래스 이름"))


def format_translations(translations) -> str:
    """워커 done 이벤트의 translations 목록을 종류별 '원문 → 번역' 문구로. 없으면 빈 문자열."""
    rows = [t for t in translations or [] if isinstance(t, dict)
            and isinstance(t.get("source"), str) and isinstance(t.get("text"), str)]
    if not rows:
        return ""
    lines = ["한국어 입력을 아래처럼 영어로 바꿔서 모든 단계(검출, 인페인팅, 검증)에 썼습니다.", ""]
    for kind, title in _TRANSLATION_GROUPS:
        group = [t for t in rows if t.get("kind") == kind]
        if group:
            lines += [f"[{title}]"] + [f"  {t['source']} → {t['text']}" for t in group] + [""]
    lines += ["Run도 이 번역을 그대로 씁니다 (출력 폴더의 translations.json에 저장).",
              "번역이 틀렸으면 그 항목을 영어로 직접 입력한 뒤 다시 Preview하세요."]
    return "\n".join(lines)


def format_references(references) -> str:
    """워커 done 이벤트의 references 목록을 항목별 '파일 → 설명' 문구로. 없으면 빈 문자열."""
    rows = [r for r in references or [] if isinstance(r, dict)
            and isinstance(r.get("file"), str) and isinstance(r.get("text"), str)]
    if not rows:
        return ""
    lines = ["참고 사진을 VLM이 아래처럼 설명했고, 각 항목의 후보 값에 더해 썼습니다.", ""]
    axes = list(dict.fromkeys(str(r.get("axis", "")) for r in rows))
    for axis in axes:
        lines.append(f"[{axis_label(axis)}]")
        lines += [f"  {r['file']} → {r['text']}" for r in rows if str(r.get("axis", "")) == axis]
        lines.append("")
    lines += ["Run도 이 설명을 그대로 씁니다 (출력 폴더의 references.json에 저장).",
              "설명이 사진과 다르면 그 항목 칸에 원하는 문구를 직접 적으세요."]
    return "\n".join(lines)


def format_export_message(event: dict) -> str:
    if "copied" not in event:
        return "내보내기가 완료되었습니다."
    msg = (f"내보내기 완료: {event['copied']}쌍 복사, "
           f"이미 있어 건너뜀 {event.get('skipped_existing', 0)}쌍")
    unaccepted = event.get("skipped_unaccepted", 0)
    if isinstance(unaccepted, int) and unaccepted > 0:
        msg += f", Manifest에 승인 기록이 없어 건너뜀 {unaccepted}쌍"
    return msg
