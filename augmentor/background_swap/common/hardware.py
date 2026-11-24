"""PC(GPU)마다 다른 값: 모델 적재 방식과 처리 속도 (torch 비의존).

- load_plan: GPU 메모리(VRAM) 크기로 모델 적재 방식을 고른다. 워커가 모델을 올릴 때 쓴다.
- 속도: 앱을 켤 때 워커의 probe 명령이 GPU 행렬곱 속도(TFLOPS)를 재고, 기준 GPU와의 비율로
  예상 시간을 맞춘다. Preview/Run이 끝나면 실제로 걸린 단계별 속도를 저장해 그 뒤로는 그 값을 쓴다.
- 저장 위치: ~/.config/dataset_helper/hardware_profile.json (GPU 이름별)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from background_swap.common.store import atomic_write_text

PROFILE_PATH = Path.home() / ".config" / "dataset_helper" / "hardware_profile.json"

# GPU the estimate.py constants were measured on, and its probe score (fp16 4096x4096 matmul)
REFERENCE_GPU = "NVIDIA GeForce RTX 4060 Laptop GPU"
REFERENCE_TFLOPS = 24.8  # measured 2026-11-15 (3 runs: 24.73-24.85)

_FACTOR_RANGE = (0.1, 20.0)
_SMOOTHING = 0.5          # weight of a new measurement (averaged with the previous value)


@dataclass(frozen=True)
class LoadPlan:
    """모델 적재 방식. 메모리가 작을수록 느리지만 덜 쓰는 방식을 고른다."""
    llm_load: str          # "fp16" | "4bit"
    vlm_load: str          # "fp16" | "4bit"
    inpaint_offload: str   # "none" (all on GPU) | "model" (per component CPU<->GPU) | "sequential" (least VRAM, slowest)

    def describe(self) -> str:
        offload = {"none": "전부 GPU", "model": "일부 CPU 오프로드",
                   "sequential": "순차 CPU 오프로드(느림)"}[self.inpaint_offload]
        return f"LLM {self.llm_load}, VLM {self.vlm_load}, 인페인팅 {offload}"


# Measured VRAM (docs/superpowers/notes/2026-11-06-vram-spike.md): LLM fp16 peak 5.8GB, VLM fp16 7.3GB,
# SDXL (cpu offload) 5.3-5.7GB. Full-GPU SDXL weights alone are ~7GB, so leave a wide margin.
LLM_FP16_MIN_GB = 7.5
VLM_FP16_MIN_GB = 11.0
INPAINT_FULL_GPU_MIN_GB = 15.0
INPAINT_MODEL_OFFLOAD_MIN_GB = 7.0


def load_plan(vram_gb: float | None, can_4bit: bool = True) -> LoadPlan:
    """can_4bit=False: bitsandbytes가 없는 설치(Small)라 4bit를 못 쓴다 → 항상 fp16."""
    v = float(vram_gb or 0.0)
    return LoadPlan(
        llm_load="fp16" if v >= LLM_FP16_MIN_GB or not can_4bit else "4bit",
        vlm_load="fp16" if v >= VLM_FP16_MIN_GB or not can_4bit else "4bit",
        inpaint_offload=("none" if v >= INPAINT_FULL_GPU_MIN_GB
                         else "model" if v >= INPAINT_MODEL_OFFLOAD_MIN_GB else "sequential"),
    )


def _clip(x: float) -> float:
    return min(max(float(x), _FACTOR_RANGE[0]), _FACTOR_RANGE[1])


@dataclass
class HardwareProfile:
    gpu_name: str = ""
    vram_gb: float = 0.0
    tflops: float = 0.0
    measured: dict[str, float] = field(default_factory=dict)   # per-stage ratio (measured / reference estimate)

    @property
    def known(self) -> bool:
        return bool(self.gpu_name)

    @property
    def base_factor(self) -> float:
        """기준 GPU 대비 시간 배율 (probe 값). 모르면 1."""
        if self.tflops <= 0 or REFERENCE_TFLOPS <= 0:
            return 1.0
        return _clip(REFERENCE_TFLOPS / self.tflops)

    def factor(self, stage: str) -> float:
        return self.measured.get(stage, self.base_factor)

    @property
    def load(self) -> LoadPlan:
        return load_plan(self.vram_gb)

    def describe(self) -> str:
        if not self.known:
            return "GPU 확인 중 (또는 CUDA GPU 없음) — 기준 PC 값으로 계산합니다"
        speed = ("실제 실행으로 측정한 속도" if self.measured
                 else f"GPU 속도 측정값 기준 (기준 PC의 {1 / self.base_factor:.1f}배 속도)")
        return f"{self.gpu_name}, VRAM {self.vram_gb:.1f}GB · {speed}\n  모델 적재: {self.load.describe()}"


def _read(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _from_entry(name: str, entry) -> HardwareProfile:
    if not isinstance(entry, dict):
        return HardwareProfile(gpu_name=name)
    measured = entry.get("measured") if isinstance(entry.get("measured"), dict) else {}
    try:
        return HardwareProfile(
            gpu_name=name, vram_gb=float(entry.get("vram_gb") or 0), tflops=float(entry.get("tflops") or 0),
            measured={str(k): _clip(v) for k, v in measured.items()
                      if isinstance(v, (int, float)) and v > 0})
    except (TypeError, ValueError):
        return HardwareProfile(gpu_name=name)


def load_profile(path: Path = PROFILE_PATH, gpu_name: str | None = None) -> HardwareProfile:
    """저장된 프로필. gpu_name이 없으면 마지막으로 쓴 GPU."""
    data = _read(path)
    name = gpu_name or data.get("last_gpu")
    gpus = data.get("gpus") if isinstance(data.get("gpus"), dict) else {}
    if not isinstance(name, str) or not name:
        return HardwareProfile()
    return _from_entry(name, gpus.get(name))


def save_profile(profile: HardwareProfile, path: Path = PROFILE_PATH) -> None:
    if not profile.known:
        return
    data = _read(path)
    gpus = data.get("gpus") if isinstance(data.get("gpus"), dict) else {}
    gpus[profile.gpu_name] = {"vram_gb": profile.vram_gb, "tflops": profile.tflops,
                              "measured": dict(profile.measured)}
    data.update(last_gpu=profile.gpu_name, gpus=gpus)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def merge_probe(path: Path, gpu_name: str, vram_gb: float, tflops: float) -> HardwareProfile:
    """probe 결과를 저장된 실측 배율과 합친다 (같은 GPU의 실측값은 유지)."""
    stored = load_profile(path, gpu_name)
    return HardwareProfile(gpu_name=gpu_name, vram_gb=float(vram_gb), tflops=float(tflops),
                           measured=dict(stored.measured))


def record_speed(profile: HardwareProfile, ratios: dict[str, float]) -> HardwareProfile:
    """실행에서 잰 단계별 배율(실측 / 기준 추정)을 반영한 새 프로필."""
    measured = dict(profile.measured)
    for stage, ratio in ratios.items():
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            continue
        old = measured.get(stage)
        measured[stage] = _clip(ratio if old is None else old + _SMOOTHING * (ratio - old))
    return HardwareProfile(profile.gpu_name, profile.vram_gb, profile.tflops, measured)
