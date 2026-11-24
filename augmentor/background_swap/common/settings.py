"""Background Swap 설정 (GUI가 생성, run_settings.json으로 워커에 전달)."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

# See docs/superpowers/notes/2026-11-06-vram-spike.md
MAX_RESOLUTION_PX_DEFAULT = 1280

DEFAULT_SCENE_AXES: dict[str, list[str]] = {
    "venue": [
        "school gymnasium",
        "convention center exhibition hall",
        "university auditorium",
        "warehouse-style arena",
        "sports hall",
    ],
    "lighting": [
        "cool fluorescent lights",
        "warm LED lighting",
        "stage spotlights",
        "dim indoor lighting",
        "daylight from high windows",
    ],
    "surroundings_floor": [
        "grey carpet",
        "polished concrete",
        "wooden court floor",
        "vinyl floor mats",
    ],
    "context": [
        "spectators in bleachers",
        "sponsor banners on walls",
        "equipment tables and laptops",
        "empty hall",
    ],
    "camera": [
        "smartphone photo with noise",
        "clean DSLR photo",
        "slight motion blur",
    ],
}


# Display names for scene axes (config keys stay English)
SCENE_AXIS_LABELS: dict[str, str] = {
    "venue": "장소",
    "lighting": "조명",
    "surroundings_floor": "바닥",
    "context": "주변 풍경",
    "camera": "사진 느낌",
}


def axis_label(name: str) -> str:
    return SCENE_AXIS_LABELS.get(name, name)


@dataclass
class KeepItem:
    name: str
    method: Literal["color", "text"]
    required: bool = False
    prompt: str = ""
    hsv_lower: list[int] | None = None
    hsv_upper: list[int] | None = None
    h_tol: int = 10
    s_tol: int = 60
    v_tol: int = 60
    largest_only: bool = True
    convex_hull: bool = True
    min_area_ratio: float = 0.05


@dataclass
class ModelIds:
    segmenter: str = "facebook/sam2.1-hiera-small"
    detector: str = "IDEA-Research/grounding-dino-tiny"
    llm: str = "Qwen/Qwen2.5-3B-Instruct"
    inpainter: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    vlm: str = "Qwen/Qwen2.5-VL-3B-Instruct"


# Model size presets. Large: default for 8GB GPUs. Small: for ~4GB GPUs or small disks
# (SD1.5 inpainting at 512px, Qwen2.5-1.5B, SAM2.1-tiny, no VLM check, no bitsandbytes/peft needed).
MODEL_PRESETS: dict[str, ModelIds] = {
    "large": ModelIds(),
    "small": ModelIds(segmenter="facebook/sam2.1-hiera-tiny",
                      llm="Qwen/Qwen2.5-1.5B-Instruct",
                      inpainter="stable-diffusion-v1-5/stable-diffusion-inpainting"),
}
PRESET_USE_VLM = {"large": True, "small": False}


def model_preset(models: ModelIds) -> str:
    """"small" | "large" | "custom"(고급 설정에서 모델 이름을 직접 바꾼 경우). VLM 이름은 보지 않는다."""
    for name, preset in MODEL_PRESETS.items():
        if all(getattr(models, f) == getattr(preset, f)
               for f in ("segmenter", "detector", "llm", "inpainter")):
            return name
    return "custom"


def is_sdxl(model_id: str) -> bool:
    return "xl" in (model_id or "").lower()


@dataclass
class BackgroundSwapSettings:
    images_dir: Path | None = None
    labels_dir: Path | None = None
    output_dir: Path | None = None
    class_names_file: Path | None = None
    frame_stride: int = 1
    keep_items: list[KeepItem] = field(default_factory=list)
    num_variants: int = 3
    max_attempts: int = 3
    seed: int = 0
    use_llm: bool = True
    use_vlm: bool = True
    # Fast mode: speed-up LoRA on the inpainter (several times faster, slightly lower quality)
    fast_mode: bool = False
    scene_axes: dict[str, list[str]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_SCENE_AXES))
    # Scene axes unchecked in the GUI. Their values stay in scene_axes; only the run skips them.
    disabled_scene_axes: list[str] = field(default_factory=list)
    # Optional reference photos per scene axis; the VLM describes the matching part and adds it to the candidates.
    reference_images: dict[str, list[Path]] = field(default_factory=dict)
    resolution_mode: Literal["auto", "manual"] = "auto"
    max_resolution_px: int = MAX_RESOLUTION_PX_DEFAULT
    manual_resolution_px: int = 1024
    small_box_px: int = 32
    dilate_px: int = 6
    shadow_extend_ratio: float = 0.15
    feather_px: int = 4
    min_inpaint_ratio: float = 0.15
    min_change: float = 20.0
    allow_non_train: bool = False
    models: ModelIds = field(default_factory=ModelIds)


_PATH_FIELDS = ("images_dir", "labels_dir", "output_dir", "class_names_file")

# Fields that affect results (all but the 3 path/guard fields). disabled_scene_axes is not compared
# because effective_settings already drops those axes from scene_axes before comparing.
_NON_GENERATION = ("output_dir", "allow_non_train", "class_names_file", "disabled_scene_axes")
GENERATION_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(BackgroundSwapSettings) if f.name not in _NON_GENERATION
)


def settings_to_dict(s: BackgroundSwapSettings) -> dict:
    d: dict = {}
    for f in fields(BackgroundSwapSettings):
        v = getattr(s, f.name)
        if f.name in _PATH_FIELDS:
            d[f.name] = None if v is None else str(v)
        elif f.name == "keep_items":
            d[f.name] = [{kf.name: copy.deepcopy(getattr(k, kf.name)) for kf in fields(KeepItem)}
                         for k in v]
        elif f.name == "models":
            d[f.name] = {mf.name: getattr(v, mf.name) for mf in fields(ModelIds)}
        elif f.name == "scene_axes":
            d[f.name] = {k: list(vals) for k, vals in v.items()}
        elif f.name == "reference_images":
            d[f.name] = {k: [str(p) for p in paths] for k, paths in v.items() if paths}
        else:
            d[f.name] = v
    return d


def _filter_keys(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def settings_from_dict(d: dict) -> BackgroundSwapSettings:
    kwargs = _filter_keys(BackgroundSwapSettings, d)
    for name in _PATH_FIELDS:
        if kwargs.get(name) is not None:
            kwargs[name] = Path(kwargs[name])
    if "keep_items" in kwargs:
        kwargs["keep_items"] = [
            k if isinstance(k, KeepItem) else KeepItem(**_filter_keys(KeepItem, k))
            for k in (kwargs["keep_items"] or [])
        ]
    if "models" in kwargs:
        m = kwargs["models"]
        kwargs["models"] = m if isinstance(m, ModelIds) else ModelIds(**_filter_keys(ModelIds, m or {}))
    if "scene_axes" in kwargs:
        kwargs["scene_axes"] = {k: list(v) for k, v in (kwargs["scene_axes"] or {}).items()}
    if "reference_images" in kwargs:
        kwargs["reference_images"] = {str(k): [Path(p) for p in (v or [])]
                                      for k, v in (kwargs["reference_images"] or {}).items()}
    if "disabled_scene_axes" in kwargs:
        kwargs["disabled_scene_axes"] = [str(n) for n in (kwargs["disabled_scene_axes"] or [])]
    return BackgroundSwapSettings(**kwargs)


def save_settings(s: BackgroundSwapSettings, path: Path) -> None:
    """같은 디렉터리의 임시 파일에 쓴 뒤 os.replace로 원자적 교체."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings_to_dict(s), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_settings(path: Path) -> BackgroundSwapSettings:
    with open(path, "r", encoding="utf-8") as f:
        return settings_from_dict(json.load(f))


def active_scene_axes(s: BackgroundSwapSettings) -> dict[str, list[str]]:
    """체크된 배경 조합 항목만."""
    off = set(s.disabled_scene_axes)
    return {k: list(v) for k, v in s.scene_axes.items() if k not in off}


def active_reference_images(s: BackgroundSwapSettings) -> dict[str, list[Path]]:
    """체크된 배경 조합 항목의 참고 사진만 (사진이 있는 항목만)."""
    active = active_scene_axes(s)
    return {k: list(v) for k, v in s.reference_images.items() if k in active and v}


def effective_settings(s: BackgroundSwapSettings) -> BackgroundSwapSettings:
    """실행에 실제로 쓰이는 설정: 체크를 끈 배경 조합 항목(과 그 참고 사진)을 뺀 사본."""
    return dataclasses.replace(s, scene_axes=active_scene_axes(s), disabled_scene_axes=[],
                               reference_images=active_reference_images(s))


def generation_diff(a: BackgroundSwapSettings, b: BackgroundSwapSettings) -> list[str]:
    """결과에 영향을 주는 필드 중 값이 다른 것의 이름 (중첩 dataclass는 값 비교).

    체크를 끈 배경 조합 항목은 결과에 영향이 없으므로 비교에서 뺀다.
    """
    a, b = effective_settings(a), effective_settings(b)
    return [n for n in GENERATION_FIELDS if getattr(a, n) != getattr(b, n)]


def _resolved(p: Path) -> Path:
    return Path(p).resolve()


def validate_settings(s: BackgroundSwapSettings) -> list[str]:
    errs: list[str] = []

    for attr, label in (("images_dir", "이미지 폴더"), ("labels_dir", "라벨 폴더")):
        p = getattr(s, attr)
        if p is None:
            errs.append(f"{label}가 지정되지 않았습니다.")
        elif not Path(p).is_dir():
            errs.append(f"{label}가 존재하지 않습니다: {p}")
    if s.output_dir is None:
        errs.append("출력 폴더가 지정되지 않았습니다.")
    else:
        out = _resolved(s.output_dir)
        for attr, label in (("images_dir", "이미지 폴더"), ("labels_dir", "라벨 폴더")):
            p = getattr(s, attr)
            if p is not None and _resolved(p) == out:
                errs.append(f"출력 폴더는 {label}와 달라야 합니다: {s.output_dir}")

    if not 1 <= s.num_variants <= 20:
        errs.append("변형 수(num_variants)는 1~20 이어야 합니다.")
    if not 1 <= s.max_attempts <= 10:
        errs.append("최대 시도 횟수(max_attempts)는 1~10 이어야 합니다.")
    if not 1 <= s.frame_stride <= 100:
        errs.append("프레임 간격(frame_stride)은 1~100 이어야 합니다.")

    for k in s.keep_items:
        if k.method == "color":
            if not k.hsv_lower or not k.hsv_upper:
                errs.append(f"유지 항목 '{k.name}': color 방식은 HSV 범위(hsv_lower/hsv_upper)가 필요합니다.")
        elif k.method == "text":
            if not k.prompt.strip():
                errs.append(f"유지 항목 '{k.name}': text 방식은 프롬프트가 비어 있으면 안 됩니다.")
        else:
            errs.append(f"유지 항목 '{k.name}': 알 수 없는 방식입니다: {k.method}")

    refs = active_reference_images(s)
    for axis, paths in refs.items():
        missing = [str(p) for p in paths if not Path(p).is_file()]
        if missing:
            errs.append(f"배경 조합 '{axis_label(axis)}'의 참고 사진이 없습니다: " + ", ".join(missing))

    if s.fast_mode and not is_sdxl(s.models.inpainter):
        errs.append("빠른 모드는 SDXL 인페인팅 모델(Large)에서만 쓸 수 있습니다. "
                    "Small 모델은 이미 빠르니 빠른 모드를 끄세요.")

    active = active_scene_axes(s)
    if not active:
        errs.append("배경 조합 항목을 1개 이상 체크하세요.")
    for axis, values in active.items():
        if not values and not refs.get(axis):
            errs.append(f"배경 조합 '{axis_label(axis)}'에는 후보 값이나 참고 사진이 1개 이상 필요합니다 "
                        "(안 쓰려면 체크를 끄세요).")
    return errs
