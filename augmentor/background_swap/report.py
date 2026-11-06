"""실행 리포트: 상태/사유 집계, 장면 축 분포, Grounding DINO 클래스별 recall, 단계별 소요 시간."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .domain import ACCEPTED, REJECTED, SKIPPED, Detection
from .store import atomic_write_text, read_jsonl
from .validate import iou

RECALL_IOU = 0.5
NOTE = "GDINO 미검출은 새 객체가 없다는 뜻이 아닙니다."


def class_recall(label_boxes: list[tuple[int, tuple[int, int, int, int]]],
                 dets: list[Detection], class_names: list[str]) -> dict[str, list[int]]:
    """클래스별 [hits, total]. 라벨이 같은 이름의 검출과 IoU >= 0.5면 hit. 범위 밖 cls는 무시."""
    out = {name: [0, 0] for name in class_names}
    for cls, box in label_boxes:
        if not 0 <= cls < len(class_names):
            continue
        name = class_names[cls]
        out[name][1] += 1
        if any(d.label == name and iou(box, d.box) >= RECALL_IOU for d in dets):
            out[name][0] += 1
    return out


def merge_recall(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, list[int]]:
    out = {k: [v[0], v[1]] for k, v in a.items()}
    for k, v in b.items():
        cur = out.setdefault(k, [0, 0])
        cur[0] += v[0]
        cur[1] += v[1]
    return out


def _scene_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    axes: dict[str, Counter] = {}
    for r in records:
        for axis, value in r["scene"].items():
            axes.setdefault(axis, Counter())[value] += 1
    return {axis: dict(c) for axis, c in axes.items()}


def build_report(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    records = read_jsonl(output_dir / "Manifest.jsonl")
    status_counts = Counter(r.get("status") for r in records)
    reason_counts: Counter = Counter()
    for r in records:
        if r.get("status") in (REJECTED, SKIPPED):
            reason_counts.update(r.get("reasons") or [])
    scened = [r for r in records
              if r.get("status") in (ACCEPTED, REJECTED) and r.get("scene")]
    report: dict = {
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "scene_counts": {
            "all": _scene_counts(scened),
            "accepted": _scene_counts([r for r in scened if r["status"] == ACCEPTED]),
        },
    }
    recall_path = output_dir / "recall.json"
    if recall_path.exists():
        raw = json.loads(recall_path.read_text(encoding="utf-8"))
        report["class_recall"] = {
            name: {"hits": hits, "total": total,
                   "recall": round(hits / total, 4) if total else None}
            for name, (hits, total) in raw.items()}
    timings_path = output_dir / "timings.json"
    if timings_path.exists():
        report["stage_seconds"] = json.loads(timings_path.read_text(encoding="utf-8"))
    report["note"] = NOTE
    return report


def write_report(output_dir: Path) -> dict:
    report = build_report(output_dir)
    atomic_write_text(Path(output_dir) / "report.json",
                      json.dumps(report, ensure_ascii=False, indent=2))
    return report
