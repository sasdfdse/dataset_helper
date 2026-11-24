"""크래시에 안전한 파일 저장 헬퍼 (원자적 쓰기, JSON Lines, 마스크 PNG)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Run outputs reset_output_dir deletes (it never touches any other name).
# _preview/ is intentionally excluded.
_RUN_ARTIFACTS = (
    "items.jsonl", "masks", "jobs.jsonl", "generated", "Accepted", "Rejected",
    "Manifest.jsonl", "report.json", "run_settings.json",
    "recall.json", "timings.json",
)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """같은 폴더의 임시 파일에 쓰고 fsync 후 os.replace로 교체한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def imwrite_atomic(path: Path, img: np.ndarray, jpeg_quality: int = 95) -> None:
    """확장자 기준으로 인코딩한 뒤 원자적으로 저장. 인코딩 실패/미지원 확장자는 ValueError."""
    path = Path(path)
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    try:
        ok, buf = cv2.imencode(ext, img, params)
    except cv2.error as e:
        raise ValueError(f"cannot encode image as '{ext}': {e}") from e
    if not ok:
        raise ValueError(f"cannot encode image as '{ext}'")
    atomic_write_bytes(path, buf.tobytes())


def write_mask(path: Path, mask: np.ndarray) -> None:
    """bool 마스크를 0/255 PNG로 저장."""
    imwrite_atomic(Path(path), (np.asarray(mask).astype(bool).astype(np.uint8)) * 255)


def read_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return img > 127


def append_jsonl(path: Path, record: dict) -> None:
    """한 줄 추가 + flush/fsync. 파일이 개행으로 끝나지 않으면(잘린 줄) 먼저 개행을 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                line = "\n" + line
        f.seek(0, os.SEEK_END)
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """없으면 []. 파싱 실패/객체가 아닌 줄/빈 줄은 무시."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def reset_output_dir(output_dir: Path) -> None:
    """실행 산출물만 삭제한다. 그 외 파일(_preview/ 포함)은 보존."""
    output_dir = Path(output_dir)
    for name in _RUN_ARTIFACTS:
        p = output_dir / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
