"""Background Swap CLI 워커: GUI가 서브프로세스로 실행한다.

stdout에는 JSON Lines만 출력한다(라이브러리 경고/로그는 stderr로 보낸다).
종료 코드: 0 성공, 1 기타 예외, 2 설정 오류, 3 non-train 거부,
4 기존 출력과 설정 불일치, 5 의존성/사전 점검 실패, 6 GPU OOM.
명령: run, preview, export, probe(GPU 확인: done 이벤트에 gpu/vram_gb/tflops).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, TextIO

from background_swap.dataset.classes import find_class_names_file, load_class_names
from background_swap.common.domain import GpuOutOfMemory
from background_swap.dataset.export import NonTrainTargetError, export_accepted
from background_swap.dataset.inputs import looks_non_train
from background_swap.common.progress import JsonLinesProgress
from background_swap.common.settings import (BackgroundSwapSettings, effective_settings, generation_diff,
                                             load_settings, validate_settings)
from background_swap.pipeline.workflow import BackgroundSwapWorkflow, ModelFactory, run_preview
from background_swap.stages.references import CACHE_NAME as REFERENCE_CACHE_NAME
from background_swap.stages.references import ReferencePhotoError, describe_references, pending_references
from background_swap.stages.translate import CACHE_NAME, TranslationError, pending_sources, translate_inputs

EXIT_OK, EXIT_OTHER, EXIT_SETTINGS, EXIT_NON_TRAIN, EXIT_MISMATCH, EXIT_DEPS, EXIT_OOM = (
    0, 1, 2, 3, 4, 5, 6)

OOM_MESSAGE = "VRAM 부족 — 고급 설정에서 해상도를 낮추거나 더 작은 모델을 사용하세요."
NO_CLASS_NAMES_MESSAGE = "클래스 이름 파일이 없어 새 객체 검사를 건너뜁니다."
DEPS_HINT = "make venv (Large) 또는 make venv-small (Small)"

FactoryBuilder = Callable[[BackgroundSwapSettings, list], ModelFactory]


class _Fail(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m background_swap",
                                description="Background Swap 워커")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="전체 실행(이어하기 지원)")
    run.add_argument("--settings", required=True, type=Path)
    prev = sub.add_parser("preview", help="무작위 n장 미리보기")
    prev.add_argument("--settings", required=True, type=Path)
    prev.add_argument("--n", type=int, default=3)
    prev.add_argument("--seed", type=int, default=0)
    exp = sub.add_parser("export", help="Accepted 결과를 학습 폴더로 내보내기")
    exp.add_argument("--settings", required=True, type=Path)
    exp.add_argument("--images-dir", required=True, type=Path)
    exp.add_argument("--labels-dir", required=True, type=Path)
    sub.add_parser("probe", help="GPU 이름/VRAM/속도 확인 (모델은 올리지 않음)")
    return p


def _probe() -> dict:
    try:
        from background_swap.models.probe import probe_gpu
    except ImportError as e:
        raise _Fail(EXIT_DEPS, f"필수 패키지를 불러올 수 없습니다 ({e}). 다음을 실행하세요: {DEPS_HINT}")
    return probe_gpu()


def _load(path: Path) -> BackgroundSwapSettings:
    try:
        return load_settings(path)
    except Exception as e:
        raise _Fail(EXIT_SETTINGS, f"설정 파일을 읽을 수 없습니다: {path} ({type(e).__name__}: {e})")


def _check_train_only(s: BackgroundSwapSettings) -> None:
    if not s.allow_non_train and looks_non_train(Path(s.images_dir)):
        raise _Fail(EXIT_NON_TRAIN,
                    f"학습용이 아닌 폴더(val/test 등)로 보이는 입력입니다: {s.images_dir}. "
                    "정말 사용하려면 설정에서 허용 옵션을 켜세요.")


def _check_resume(s: BackgroundSwapSettings, given: Path) -> None:
    existing_path = Path(s.output_dir) / "run_settings.json"
    if not existing_path.is_file() or existing_path.resolve() == given.resolve():
        return
    diff = generation_diff(_load(existing_path), s)
    if diff:
        raise _Fail(EXIT_MISMATCH,
                    "기존 출력의 설정과 다릅니다. 달라진 항목: " + ", ".join(diff)
                    + ". 새 출력 폴더를 사용하거나 기존 설정으로 이어서 실행하세요.")


def _class_names(s: BackgroundSwapSettings, progress: JsonLinesProgress) -> list[str]:
    path = s.class_names_file or find_class_names_file(Path(s.images_dir), Path(s.labels_dir))
    names = load_class_names(Path(path)) if path else []
    if not names:
        progress.event("warning", message=NO_CLASS_NAMES_MESSAGE)
    return names


def _default_factory_builder(s: BackgroundSwapSettings, needs_translation: bool = False,
                             needs_describer: bool = False) -> FactoryBuilder:
    try:
        from background_swap.models import HFModelFactory, preflight
        # Translation needs the LLM and reference captions need the VLM, so check them even when disabled
        checked = dataclasses.replace(s, use_llm=s.use_llm or needs_translation,
                                      use_vlm=s.use_vlm or needs_describer)
        problems = preflight(checked)
    except ImportError as e:
        raise _Fail(EXIT_DEPS, f"필수 패키지를 불러올 수 없습니다 ({e}). "
                               f"다음을 실행하세요: {DEPS_HINT}")
    if problems:
        raise _Fail(EXIT_DEPS, "사전 점검 실패: " + "; ".join(problems))
    return lambda settings, class_names: HFModelFactory(settings, class_names)


def _prepare_models(s: BackgroundSwapSettings, factory_builder: FactoryBuilder | None,
                    class_names: list[str], progress: JsonLinesProgress,
                    ) -> tuple[BackgroundSwapSettings, list[str], ModelFactory, list[dict], list[dict]]:
    """한글 입력을 영어로 바꾸고 참고 사진 설명을 배경 조합에 더한 설정/클래스 이름, 그 설정으로 만든
    모델 팩토리, 적용한 번역 목록, 참고 사진 설명 목록을 돌려준다."""
    cache = Path(s.output_dir) / CACHE_NAME
    ref_cache = Path(s.output_dir) / REFERENCE_CACHE_NAME
    pending = pending_sources(s, class_names, cache)
    builder = (factory_builder if factory_builder is not None
               else _default_factory_builder(s, needs_translation=bool(pending),
                                             needs_describer=bool(pending_references(s, ref_cache))))
    factory = builder(s, class_names)
    try:
        new_s, new_names, used = translate_inputs(s, class_names, factory.translator, progress, cache)
    except TranslationError as e:
        raise _Fail(EXIT_SETTINGS, str(e))
    if used:
        factory = builder(new_s, new_names)
    try:
        new_s, references = describe_references(new_s, factory.describer, progress, ref_cache)
    except ReferencePhotoError as e:
        raise _Fail(EXIT_SETTINGS, str(e))
    return new_s, new_names, factory, [t.to_dict() for t in used], references


def _execute(args: argparse.Namespace, factory_builder: FactoryBuilder | None,
             progress: JsonLinesProgress) -> dict:
    """성공 시 done 이벤트에 덧붙일 필드를 반환. 실패는 _Fail로 알린다."""
    if args.command == "probe":
        with contextlib.redirect_stdout(sys.stderr):
            return _probe()
    s = _load(args.settings)

    if args.command == "export":
        if s.output_dir is None:
            raise _Fail(EXIT_SETTINGS, "출력 폴더가 지정되지 않았습니다.")
        try:
            r = export_accepted(Path(s.output_dir), args.images_dir, args.labels_dir,
                                s.allow_non_train)
        except NonTrainTargetError as e:
            raise _Fail(EXIT_NON_TRAIN, str(e))
        return {"copied": r.copied, "skipped_existing": r.skipped_existing,
                "skipped_unaccepted": r.skipped_unaccepted}

    errors = validate_settings(s)
    if errors:
        raise _Fail(EXIT_SETTINGS, "; ".join(errors))
    _check_train_only(s)
    if args.command == "run":
        _check_resume(s, args.settings)
    s = effective_settings(s)
    class_names = _class_names(s, progress)
    # Route stdout from heavy imports/preflight/models/workflow to stderr;
    # progress keeps writing to the real stdout it bound before this redirect.
    with contextlib.redirect_stdout(sys.stderr):
        s, class_names, factory, translations, references = _prepare_models(
            s, factory_builder, class_names, progress)
        extra = {"translations": translations} if translations else {}
        if references:
            extra["references"] = references
        if args.command == "run":
            BackgroundSwapWorkflow(s, factory, progress, class_names).run()
            return extra
        preview_dir = run_preview(s, factory, progress, class_names, args.n, args.seed)
    return {"preview_dir": str(preview_dir), **extra}


def main(argv: list[str], factory_builder: FactoryBuilder | None = None,
         stream: TextIO | None = None) -> int:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    stream = stream if stream is not None else sys.stdout
    args = _parser().parse_args(argv)
    progress = JsonLinesProgress(stream)

    def fail(code: int, message: str) -> int:
        progress.event("error", code=code, message=message)
        return code

    try:
        extra = _execute(args, factory_builder, progress)
    except _Fail as f:
        return fail(f.code, f.message)
    except GpuOutOfMemory:
        return fail(EXIT_OOM, OOM_MESSAGE)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return fail(EXIT_OTHER, f"예기치 않은 오류: {type(e).__name__}: {e}")
    progress.event("done", **extra)
    return EXIT_OK
