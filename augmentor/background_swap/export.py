"""Accepted 결과를 학습 데이터셋 폴더로 복사한다."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .inputs import looks_non_train


class NonTrainTargetError(ValueError):
    pass


@dataclass
class ExportResult:
    copied: int
    skipped_existing: int


def export_accepted(output_dir: Path, images_dir: Path, labels_dir: Path,
                    allow_non_train: bool) -> ExportResult:
    """Accepted/{images,labels}의 (이미지, 라벨) 쌍을 대상 폴더로 복사한다.

    - 라벨이 없는 이미지는 쌍이 아니므로 무시한다(집계에도 넣지 않음).
    - 이미지 또는 라벨 중 하나라도 대상에 이미 있으면 쌍 전체를 건너뛴다(덮어쓰지 않음).
    - 파일 내용만 복사한다(메타데이터/권한 제외).
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
        shutil.copyfile(img, dst_img)
        shutil.copyfile(lbl, dst_lbl)
        copied += 1
    return ExportResult(copied, skipped)
