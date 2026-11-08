"""단계별 일괄 처리 + 라운드 재시도 + 재개 가능한 Background Swap 오케스트레이터 (torch 비의존).

모델은 ModelFactory의 컨텍스트 매니저로만 얻는다. 한 단계에서 한 모델 계열만 열고,
그 단계의 모든 항목을 처리한 뒤 닫는다(종료 시 GPU 해제는 팩토리 책임).

재개 기준 (모두 디스크 산출물):
- masks: masks/{stem}.json 존재
- prompts: jobs.jsonl에 (stem, n, k) 존재
- inpaint: generated/{stem}_bgswap{n}_a{k}.png 존재
- validate: Manifest에 (stem, n, k) 레코드 존재
"""
from __future__ import annotations

import dataclasses
import json
import random
import shutil
import time
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from core.label import BBox, read_labels

from background_swap.stages.composite import draw_boxes
from background_swap.common.domain import (ACCEPTED, EXHAUSTED, REJECTED, SKIPPED, Detection, ImageItem, Job,
                     MaskSet)
from background_swap.stages.inpaint import Inpainter, run_inpaint
from background_swap.dataset.inputs import collect_items, load_image_bgr
from background_swap.stages.masks import Segmenter, TextDetector, bbox_to_xyxy, build_mask_set
from background_swap.common.progress import Progress
from background_swap.dataset.report import class_recall, merge_recall, write_report
from background_swap.stages.prompts import (PromptGenerator, PromptRequest, TemplatePromptGenerator, derive_seed,
                      finalize_prompt, negative_prompt, sample_scene)
from background_swap.common.settings import BackgroundSwapSettings
from background_swap.common.store import (append_jsonl, atomic_write_bytes, atomic_write_text, imwrite_atomic,
                    read_jsonl, read_mask, write_mask)
from background_swap.stages.validate import VLM_PROMPT, ImageValidator, combine, find_new_objects, rule_checks

PROMPT_BATCH = 8
VLM_BATCH = 4
NEW_OBJ_BOX_THRESHOLD = 0.35
NEW_OBJ_TEXT_THRESHOLD = 0.25
MASK_NAMES = ("object_core", "object_protect", "keep", "inpaint")


class ModelFactory(Protocol):
    """각 메서드는 컨텍스트 매니저. 종료 시 모델을 해제(GPU 메모리 반환)한다."""

    def mask_models(self) -> AbstractContextManager[tuple[Segmenter, TextDetector]]: ...
    def prompt_model(self) -> AbstractContextManager[PromptGenerator]: ...
    def inpainter(self) -> AbstractContextManager[Inpainter]: ...
    def detector(self) -> AbstractContextManager[TextDetector]: ...
    def validator(self) -> AbstractContextManager[ImageValidator]: ...


UNSUPPORTED_LABEL_FORMAT = "unsupported_label_format"


def _read_labels_strict(label_path: Path) -> list[BBox] | None:
    """core.label.read_labels로 읽되, 버려진 줄(5필드가 아닌 줄)이 있거나 파싱 예외면 None.

    read_labels는 기존 augmentor와 공유하므로 동작을 바꾸지 않고, 비어 있지 않은 줄 수와
    파싱된 박스 수를 비교해 지원하지 않는 형식(신뢰도 열, 세그멘테이션 폴리곤, '0.0' 클래스 등)을 걸러낸다.
    """
    try:
        boxes = read_labels(label_path)
        with open(label_path) as f:
            lines = sum(1 for line in f if line.strip())
    except ValueError:   # int('0.0'), float 변환 실패, UnicodeDecodeError
        return None
    return boxes if len(boxes) == lines else None


def _key(stem: str, variant, attempt) -> tuple:
    return (stem, variant, attempt)


def _check_count(stage: str, what: str, expected: int, got: list) -> list:
    """모델 배치 출력 수가 입력 수와 다르면 조용히 잘라내지 않고 실패한다."""
    if len(got) != expected:
        raise ValueError(f"{stage} stage: {what} returned {len(got)} results "
                         f"for {expected} inputs")
    return got


class BackgroundSwapWorkflow:
    def __init__(self, settings: BackgroundSwapSettings, factory: ModelFactory,
                 progress: Progress, class_names: list[str],
                 only_stems: list[str] | None = None) -> None:
        self.settings = settings
        self.factory = factory
        self.progress = progress
        self.class_names = list(class_names)
        self.only_stems = None if only_stems is None else set(only_stems)
        self.out = Path(settings.output_dir)
        self._items: dict[str, ImageItem] = {}
        self._mask_meta: dict[str, dict] = {}
        # Manifest 색인 (파일과 동기화: 추가할 때마다 갱신)
        self._skipped_stems: set[str] = set()
        self._attempted: set[tuple] = set()       # (stem, n, k)
        self._accepted: set[tuple] = set()        # (stem, n)
        self._exhausted: set[tuple] = set()       # (stem, n)

    # ── 경로 헬퍼 ──────────────────────────────────────────────────

    @property
    def manifest_path(self) -> Path:
        return self.out / "Manifest.jsonl"

    @property
    def jobs_path(self) -> Path:
        return self.out / "jobs.jsonl"

    def mask_json_path(self, stem: str) -> Path:
        return self.out / "masks" / f"{stem}.json"

    def mask_png_path(self, stem: str, name: str) -> Path:
        return self.out / "masks" / f"{stem}_{name}.png"

    def generated_path(self, stem: str, n: int, k: int) -> Path:
        return self.out / "generated" / f"{stem}_bgswap{n}_a{k}.png"

    def accepted_image_path(self, stem: str, n: int) -> Path:
        return self.out / "Accepted" / "images" / f"{stem}_bgswap{n}.jpg"

    def accepted_label_path(self, stem: str, n: int) -> Path:
        return self.out / "Accepted" / "labels" / f"{stem}_bgswap{n}.txt"

    def accepted_preview_path(self, stem: str, n: int) -> Path:
        return self.out / "Accepted" / "previews" / f"{stem}_bgswap{n}.jpg"

    def rejected_path(self, stem: str, n: int, k: int) -> Path:
        return self.out / "Rejected" / f"{stem}_bgswap{n}_a{k}.jpg"

    # ── 실행 ───────────────────────────────────────────────────────

    def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self._load_manifest_index()
        self._stage_inputs()
        with self._timed("masks"):
            self._stage_masks()
        active = [stem for stem in self._items
                  if stem in self._mask_meta and self._mask_meta[stem].get("skip_reason") is None]
        variants = range(1, self.settings.num_variants + 1)
        for k in range(1, self.settings.max_attempts + 1):
            targets = [(stem, n) for stem in active for n in variants
                       if (stem, n) not in self._accepted
                       and _key(stem, n, k) not in self._attempted]
            if not targets:
                # 이 라운드는 이미 끝났다(재개). 이전 실행이 라운드 k 이후에 중단됐을 수 있으므로
                # 종료하지 않고 다음 라운드로 넘어간다 — 대상이 없으면 아무 모델도 열지 않는다.
                continue
            with self._timed("prompts"):
                jobs = self._stage_prompts(targets, k)
            with self._timed("inpaint"):
                self._stage_inpaint(jobs)
            with self._timed("validate"):
                self._stage_validate(jobs)
        for stem in active:
            for n in variants:
                if (stem, n) not in self._accepted and (stem, n) not in self._exhausted:
                    self._append_manifest(self._record(
                        stem, None, status=EXHAUSTED, reasons=[], checks={},
                        warnings=self._warnings(stem), variant=n))
        self.progress.stage("report", 1)
        t0 = time.perf_counter()
        write_report(self.out)
        # 리포트 자신의 소요 시간은 report.json 작성 뒤에 누적한다(다음 실행의 리포트에 반영).
        self._add_timing("report", time.perf_counter() - t0)
        self.progress.advance()

    # ── 단계별 소요 시간 ───────────────────────────────────────────

    @property
    def timings_path(self) -> Path:
        return self.out / "timings.json"

    def _add_timing(self, stage: str, seconds: float) -> None:
        timings: dict[str, float] = {}
        if self.timings_path.exists():
            timings = json.loads(self.timings_path.read_text(encoding="utf-8"))
        timings[stage] = timings.get(stage, 0.0) + seconds
        atomic_write_text(self.timings_path, json.dumps(timings, ensure_ascii=False))

    @contextmanager
    def _timed(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._add_timing(stage, time.perf_counter() - t0)

    # ── Manifest ───────────────────────────────────────────────────

    def _load_manifest_index(self) -> None:
        for r in read_jsonl(self.manifest_path):
            self._index(r)

    def _index(self, r: dict) -> None:
        stem, status = r.get("stem"), r.get("status")
        n, k = r.get("variant"), r.get("attempt")
        if status == SKIPPED:
            self._skipped_stems.add(stem)
            return
        if status == EXHAUSTED:
            self._exhausted.add((stem, n))
            return
        if status in (ACCEPTED, REJECTED):
            self._attempted.add(_key(stem, n, k))
            if status == ACCEPTED:
                self._accepted.add((stem, n))

    def _append_manifest(self, record: dict) -> None:
        append_jsonl(self.manifest_path, record)
        self._index(record)

    @staticmethod
    def _record(stem: str, job: Job | None, *, status: str, reasons: list[str],
                checks: dict, warnings: list[str], variant: int | None = None) -> dict:
        return {
            "stem": stem,
            "variant": job.variant if job is not None else variant,
            "attempt": job.attempt if job is not None else None,
            "seed": job.seed if job is not None else None,
            "scene": dict(job.scene) if job is not None else None,
            "prompt": job.prompt if job is not None else None,
            "status": status,
            "reasons": list(reasons),
            "checks": dict(checks),
            "warnings": list(warnings),
        }

    def _record_skip(self, stem: str, reason: str, warnings: list[str]) -> None:
        if stem in self._skipped_stems:
            return
        self._append_manifest(self._record(stem, None, status=SKIPPED, reasons=[reason],
                                           checks={}, warnings=warnings))

    def _warnings(self, stem: str) -> list[str]:
        return list(self._mask_meta.get(stem, {}).get("warnings", []))

    # ── Stage 0: inputs ────────────────────────────────────────────

    def _stage_inputs(self) -> None:
        s = self.settings
        items, skipped = collect_items(Path(s.images_dir), Path(s.labels_dir), s.frame_stride)
        if self.only_stems is not None:
            items = [it for it in items if it.stem in self.only_stems]
            skipped = [(stem, r) for stem, r in skipped if stem in self.only_stems]
        self.progress.stage("inputs", len(items) + len(skipped))
        self._items = {it.stem: it for it in items}
        lines = [json.dumps({"stem": it.stem, "image_path": str(it.image_path),
                             "label_path": str(it.label_path), "width": it.width,
                             "height": it.height}, ensure_ascii=False)
                 for it in items]
        atomic_write_text(self.out / "items.jsonl", "".join(l + "\n" for l in lines))
        for stem, reason in skipped:
            self._record_skip(stem, reason, [])
        self.progress.advance(len(items) + len(skipped))

    # ── Stage 1: masks ─────────────────────────────────────────────

    def _stage_masks(self) -> None:
        todo: list[ImageItem] = []
        for it in self._items.values():
            path = self.mask_json_path(it.stem)
            if path.exists():
                self._mask_meta[it.stem] = json.loads(path.read_text(encoding="utf-8"))
            else:
                todo.append(it)
        self.progress.stage("masks", len(todo))
        need_recall = bool(self.class_names) and not self.recall_path.exists()
        if not todo:
            if need_recall:
                with self.factory.detector() as detector:
                    self._compute_recall(detector)
            return
        with self.factory.mask_models() as (segmenter, detector):
            for it in todo:
                self._make_masks(it, segmenter, detector)
                self.progress.advance()
            if need_recall:   # 같은 컨텍스트의 detector 재사용 (추가 컨텍스트 없음)
                self._compute_recall(detector)

    @property
    def recall_path(self) -> Path:
        return self.out / "recall.json"

    def _compute_recall(self, detector: TextDetector) -> None:
        """스킵되지 않은 모든 항목의 원본에 GDINO를 돌려 클래스별 [hits, total]을 recall.json에 쓴다."""
        total: dict[str, list[int]] = {name: [0, 0] for name in self.class_names}
        for it in self._items.values():
            if self._mask_meta.get(it.stem, {}).get("skip_reason", "missing") is not None:
                continue
            img = load_image_bgr(it.image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), list(self.class_names),
                                   NEW_OBJ_BOX_THRESHOLD, NEW_OBJ_TEXT_THRESHOLD)
            boxes = [(b.cls, xyxy) for b in read_labels(it.label_path)
                     if 0 <= b.cls < len(self.class_names)
                     and (xyxy := bbox_to_xyxy(b, w, h, 0.0)) is not None]
            total = merge_recall(total, class_recall(boxes, dets, self.class_names))
        atomic_write_text(self.recall_path, json.dumps(total, ensure_ascii=False))

    def _make_masks(self, it: ImageItem, segmenter: Segmenter, detector: TextDetector) -> None:
        boxes = _read_labels_strict(Path(it.label_path))
        img = load_image_bgr(it.image_path) if boxes is not None else None
        if boxes is None:
            meta = {"warnings": [], "skip_reason": UNSUPPORTED_LABEL_FORMAT}
        elif img is None:
            meta = {"warnings": [], "skip_reason": "unreadable_image"}
        else:
            ms = build_mask_set(img, boxes, self.settings, segmenter, detector)
            meta = {"warnings": list(ms.warnings), "skip_reason": ms.skip_reason}
            if ms.skip_reason is None:
                for name in MASK_NAMES:
                    write_mask(self.mask_png_path(it.stem, name), getattr(ms, name))
        if meta["skip_reason"] is not None:
            self._record_skip(it.stem, meta["skip_reason"], meta["warnings"])
        # json은 마지막에 써서 재개 기준으로 삼는다
        atomic_write_text(self.mask_json_path(it.stem), json.dumps(meta, ensure_ascii=False))
        self._mask_meta[it.stem] = meta

    def _load_masks(self, stem: str) -> MaskSet:
        m = {name: read_mask(self.mask_png_path(stem, name)) for name in MASK_NAMES}
        return MaskSet(m["object_core"], m["object_protect"], m["keep"], m["inpaint"],
                       self._warnings(stem), None)

    # ── Stage 2: prompts ───────────────────────────────────────────

    def _stage_prompts(self, targets: list[tuple[str, int]], k: int) -> list[Job]:
        existing: dict[tuple, Job] = {}
        for d in read_jsonl(self.jobs_path):
            try:
                job = Job.from_dict(d)
            except (KeyError, TypeError, ValueError):
                continue
            existing.setdefault(_key(job.stem, job.variant, job.attempt), job)

        s = self.settings
        keep_names = [ki.name for ki in s.keep_items]
        todo: list[tuple[str, int, int, dict]] = []
        for stem, n in targets:
            if _key(stem, n, k) in existing:
                continue
            seed = derive_seed(s.seed, stem, n, k)
            todo.append((stem, n, seed, sample_scene(s.scene_axes, seed)))

        self.progress.stage("prompts", len(todo))
        if todo:
            if s.use_llm:
                with self.factory.prompt_model() as gen:
                    self._generate_jobs(gen, todo, k, keep_names, existing)
            else:
                self._generate_jobs(TemplatePromptGenerator(), todo, k, keep_names, existing)
        return [existing[_key(stem, n, k)] for stem, n in targets]

    def _generate_jobs(self, gen: PromptGenerator, todo: list, k: int,
                       keep_names: list[str], existing: dict) -> None:
        neg = negative_prompt(self.class_names)
        for i in range(0, len(todo), PROMPT_BATCH):
            batch = todo[i:i + PROMPT_BATCH]
            reqs = [PromptRequest(scene=dict(scene), keep_names=list(keep_names), seed=seed)
                    for _, _, seed, scene in batch]
            raws = _check_count("prompts", "generate_batch", len(batch), gen.generate_batch(reqs))
            for (stem, n, seed, scene), req, raw in zip(batch, reqs, raws, strict=True):
                job = Job(stem=stem, variant=n, attempt=k, seed=seed, scene=dict(scene),
                          prompt=finalize_prompt(raw, req, self.class_names,
                                                 list(self.settings.scene_axes)),
                          negative_prompt=neg)
                append_jsonl(self.jobs_path, job.to_dict())
                existing[_key(stem, n, k)] = job
                self.progress.advance()

    # ── Stage 3: inpaint ───────────────────────────────────────────

    def _stage_inpaint(self, jobs: list[Job]) -> None:
        todo = [j for j in jobs if not self.generated_path(j.stem, j.variant, j.attempt).exists()]
        self.progress.stage("inpaint", len(todo))
        if not todo:
            return
        with self.factory.inpainter() as inpainter:
            for job in todo:
                orig = self._load_orig(job.stem)
                out = run_inpaint(inpainter, orig, self._load_masks(job.stem), job, self.settings)
                imwrite_atomic(self.generated_path(job.stem, job.variant, job.attempt), out)
                self.progress.advance()

    # ── Stage 4: validate ──────────────────────────────────────────

    def _load_orig(self, stem: str) -> np.ndarray:
        img = load_image_bgr(self._items[stem].image_path)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {self._items[stem].image_path}")
        return img

    def _load_generated(self, job: Job) -> np.ndarray:
        p = self.generated_path(job.stem, job.variant, job.attempt)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read generated image: {p}")
        return img

    def _labels(self, stem: str) -> list[BBox]:
        return read_labels(self._items[stem].label_path)

    def _stage_validate(self, jobs: list[Job]) -> None:
        todo = [j for j in jobs if _key(j.stem, j.variant, j.attempt) not in self._attempted]
        self.progress.stage("validate", len(todo))
        if not todo:
            return

        gdino: dict[tuple, list[Detection]] = {}
        if self.class_names:
            with self.factory.detector() as detector:
                for job in todo:
                    gdino[_key(job.stem, job.variant, job.attempt)] = self._new_objects(
                        detector, job)

        vlm: dict[tuple, str] = {}
        if self.settings.use_vlm:
            prompt = VLM_PROMPT.format(
                class_names=", ".join(self.class_names) or "objects of the labeled classes")
            with self.factory.validator() as validator:
                for i in range(0, len(todo), VLM_BATCH):
                    batch = todo[i:i + VLM_BATCH]
                    images = [cv2.cvtColor(draw_boxes(self._load_generated(j), self._labels(j.stem)),
                                           cv2.COLOR_BGR2RGB) for j in batch]
                    texts = _check_count("validate", "judge_batch", len(batch),
                                         validator.judge_batch(images, prompt))
                    for j, text in zip(batch, texts, strict=True):
                        vlm[_key(j.stem, j.variant, j.attempt)] = text

        for job in todo:
            key = _key(job.stem, job.variant, job.attempt)
            orig = self._load_orig(job.stem)
            out = self._load_generated(job)
            masks = self._load_masks(job.stem)
            reasons, checks = rule_checks(orig, out, masks.protect, masks.inpaint,
                                          self.settings.min_change)
            result = combine(reasons, checks,
                             gdino[key] if self.class_names else None,
                             vlm[key] if self.settings.use_vlm else None)
            self._write_result(job, out, result.accepted)
            self._append_manifest(self._record(
                job.stem, job, status=ACCEPTED if result.accepted else REJECTED,
                reasons=result.reasons, checks=result.checks,
                warnings=self._warnings(job.stem)))
            self.progress.advance()

    def _new_objects(self, detector: TextDetector, job: Job) -> list[Detection]:
        out = self._load_generated(job)
        h, w = out.shape[:2]
        dets = detector.detect(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), list(self.class_names),
                               NEW_OBJ_BOX_THRESHOLD, NEW_OBJ_TEXT_THRESHOLD)
        label_boxes = [xyxy for xyxy in (bbox_to_xyxy(b, w, h, 0.0) for b in self._labels(job.stem))
                       if xyxy is not None]
        return find_new_objects(dets, label_boxes, self._load_masks(job.stem).inpaint)

    def _write_result(self, job: Job, out: np.ndarray, accepted: bool) -> None:
        """파일을 먼저 쓴다 (Manifest 레코드는 호출자가 그 뒤에 추가)."""
        stem, n = job.stem, job.variant
        if accepted:
            imwrite_atomic(self.accepted_image_path(stem, n), out)
            atomic_write_bytes(self.accepted_label_path(stem, n),
                               Path(self._items[stem].label_path).read_bytes())
            imwrite_atomic(self.accepted_preview_path(stem, n),
                           draw_boxes(out, self._labels(stem)))
        else:
            imwrite_atomic(self.rejected_path(stem, n, job.attempt), out)
            if (stem, n) not in self._accepted:
                # 이전 실행이 Accepted 파일을 쓴 뒤 Manifest 기록 전에 중단됐다면 그 파일은
                # 승인 기록이 없는 고아다. 이번 판정이 거절이므로 지운다.
                for p in (self.accepted_image_path(stem, n), self.accepted_label_path(stem, n),
                          self.accepted_preview_path(stem, n)):
                    p.unlink(missing_ok=True)


def run_preview(settings: BackgroundSwapSettings, factory: ModelFactory, progress: Progress,
                class_names: list[str], n: int, rng_seed: int) -> Path:
    """무작위 n장(변형 1개, 시도 1회)을 output_dir/_preview에 새로 실행하고 그 경로를 반환."""
    preview_dir = Path(settings.output_dir) / "_preview"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_settings = dataclasses.replace(settings, output_dir=preview_dir,
                                           num_variants=1, max_attempts=1)
    items, _ = collect_items(Path(settings.images_dir), Path(settings.labels_dir),
                             settings.frame_stride)
    stems = sorted(it.stem for it in items)
    chosen = random.Random(rng_seed).sample(stems, max(0, min(n, len(stems))))
    BackgroundSwapWorkflow(preview_settings, factory, progress, class_names,
                           only_stems=chosen).run()
    return preview_dir
