"""클래스 이름 파일 탐색/로딩 (classes.txt, data.yaml, obj.names)."""
from __future__ import annotations

from pathlib import Path

import yaml

_CANDIDATE_NAMES = ("classes.txt", "data.yaml", "obj.names")


def find_class_names_file(images_dir: Path, labels_dir: Path) -> Path | None:
    folders: list[Path] = []
    for folder in (labels_dir, images_dir, labels_dir.parent, images_dir.parent):
        if folder not in folders:
            folders.append(folder)
    for folder in folders:
        for name in _CANDIDATE_NAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def load_class_names(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(text)
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, list):
                return [str(n) for n in names]
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=int)]
        except (yaml.YAMLError, ValueError, TypeError):
            return []
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]
