"""Accepted 결과를 학습 데이터셋 폴더로 복사한다."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .inputs import looks_non_train


class NonTrainTargetError(ValueError):
    pass


@dataclass
class ExportResult:
    copied: int
    skipped_existing: int


def _stage_copy(src: Path, dst: Path) -> Path:
    """src 내용을 dst와 같은 폴더의 임시 파일로 복사하고 임시 경로를 반환한다.

    mkstemp의 0600 대신 일반 복사와 같은 기본 모드(0o666 & ~umask)로 맞춘다.
    """
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _copy_pair(src_img: Path, dst_img: Path, src_lbl: Path, dst_lbl: Path) -> None:
    """이미지+라벨을 임시 파일 경유로 원자적으로 배치한다. 실패하면 반쪽 쌍을 남기지 않는다."""
    tmps: list[Path] = []
    placed_img = False
    try:
        tmps.append(_stage_copy(src_img, dst_img))
        tmps.append(_stage_copy(src_lbl, dst_lbl))
        os.replace(tmps[0], dst_img)
        placed_img = True
        os.replace(tmps[1], dst_lbl)
    except BaseException:
        for t in tmps:
            t.unlink(missing_ok=True)
        if placed_img:
            dst_img.unlink(missing_ok=True)
        raise


def export_accepted(output_dir: Path, images_dir: Path, labels_dir: Path,
                    allow_non_train: bool) -> ExportResult:
    """Accepted/{images,labels}의 (이미지, 라벨) 쌍을 대상 폴더로 복사한다.

    - 라벨이 없는 이미지는 쌍이 아니므로 무시한다(집계에도 넣지 않음).
    - 이미지 또는 라벨 중 하나라도 대상에 이미 있으면 쌍 전체를 건너뛴다(덮어쓰지 않음).
    - 파일 내용만 복사한다(메타데이터 제외, 기본 권한). 각 파일은 임시 파일 후 os.replace로
      배치하고, 쌍의 일부만 남지 않게 실패 시 정리한다.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not allow_non_train and (looks_non_train(images_dir) or looks_non_train(labels_dir)):
        raise NonTrainTargetError(
            f"학습용이 아닌 폴더(val/test 등)로 보이는 대상입니다: {images_dir}, {labels_dir}. "
            "정말 내보내려면 허용 옵션을 켜세요.")
    src_images = Path(output_dir) / "Accepted" / "images"
    src_labels = Path(output_dir) / "Accepted" / "labels"
    copied = skipped = 0
    if not src_images.is_dir():
        return ExportResult(0, 0)
    for img in sorted(src_images.iterdir()):
        lbl = src_labels / f"{img.stem}.txt"
        if not img.is_file() or not lbl.is_file():
            continue
        dst_img, dst_lbl = images_dir / img.name, labels_dir / lbl.name
        if dst_img.exists() or dst_lbl.exists():
            skipped += 1
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        _copy_pair(img, dst_img, lbl, dst_lbl)
        copied += 1
    return ExportResult(copied, skipped)
