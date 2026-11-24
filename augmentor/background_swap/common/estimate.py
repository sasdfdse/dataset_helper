"""작업 소요 시간 추정 (torch 비의존).

실행 전: 설정과 이미지 수/크기로 대략의 소요 시간을 계산한다 (첫 시도 기준, 재시도 제외).
실행 중: EtaTracker가 진행 이벤트의 실제 속도로 현재 단계 남은 시간을 계산하고, 뒤에 남은 단계는
추정치를 더한다.

아래 시간 상수는 기준 GPU(hardware.REFERENCE_GPU)에서 잰 값이다
(docs/superpowers/notes/2026-11-06-vram-spike.md). 다른 PC에서는 HardwareProfile의 단계별 배율을
곱한다: 앱 시작 때 잰 GPU 속도로 먼저 맞추고, Preview/Run을 한 번 돌리면 그 PC의 실측 속도로 바뀐다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from background_swap.common.hardware import HardwareProfile
from background_swap.common.settings import BackgroundSwapSettings, active_reference_images, is_sdxl
from background_swap.stages.inpaint import working_size
from background_swap.stages.references import MAX_PER_AXIS

SDXL_NATIVE_PX = 1024

# One SDXL inpaint (30 steps): fixed cost + per-pixel cost (measured: 512x288 7.2s, 1024x768 19.7s, 1280x720 23.7s)
INPAINT_BASE_S = 4.4
INPAINT_S_PER_PX = 1.95e-5
# SD1.5 inpaint (30 steps) measured: 512x288 4.1s, 512x384 4.5s, 768x432 6.5s, 768x576 8.0s
SD15_NATIVE_PX = 512
SD15_BASE_S = 2.2
SD15_S_PER_PX = 1.32e-5
FAST_INPAINT_RATIO = 6.9 / 18.6   # fast mode (Hyper-SD 8 steps) / default 30 steps, measured at 1024x768
FAST_LORA_LOAD_S = 14.0          # extra load time to fuse the fast-mode LoRA
MASK_S = 1.0             # SAM2 + Grounding DINO, per image
PROMPT_S = 1.8           # Qwen2.5-3B, per prompt
GDINO_CHECK_S = 0.3      # validation: new-object detection
VLM_CHECK_S = 1.3        # validation: Qwen2.5-VL verdict
REFERENCE_S = 2.0        # per reference photo (Qwen2.5-VL caption)
# per-stage model load time (warm disk cache)
LOAD_S = {"masks": 8.0, "prompts": 6.0, "inpaint": 15.0, "validate": 4.0, "vlm": 8.0}

STAGE_ORDER = ("translate", "references", "inputs", "masks", "prompts", "inpaint", "validate", "report")


def inpaint_seconds(width: int, height: int, settings: BackgroundSwapSettings) -> float:
    if not is_sdxl(settings.models.inpainter):     # Small: SD1.5 (native 512px)
        ww, wh = working_size(width, height, SD15_NATIVE_PX, settings.max_resolution_px,
                              settings.resolution_mode, settings.manual_resolution_px)
        return SD15_BASE_S + SD15_S_PER_PX * ww * wh
    ww, wh = working_size(width, height, SDXL_NATIVE_PX, settings.max_resolution_px,
                          settings.resolution_mode, settings.manual_resolution_px)
    seconds = INPAINT_BASE_S + INPAINT_S_PER_PX * ww * wh
    return seconds * FAST_INPAINT_RATIO if settings.fast_mode else seconds


def speed_key(settings: BackgroundSwapSettings, stage: str) -> str:
    """이 PC 실측 배율을 저장/조회하는 키: 단계 + 그 단계의 모델 (Small/Large 속도를 따로 배운다)."""
    m = settings.models
    model = {"masks": m.segmenter, "prompts": m.llm,
             "inpaint": m.inpainter + ("+fast" if settings.fast_mode else ""),
             "validate": m.vlm if settings.use_vlm else m.detector,
             "references": m.vlm}.get(stage, "")
    return f"{stage}|{model}"


@dataclass(frozen=True)
class StageCost:
    load_s: float
    per_item_s: float
    count: int

    @property
    def total_s(self) -> float:
        return (self.load_s + self.per_item_s * self.count) if self.count else 0.0


@dataclass(frozen=True)
class Plan:
    """첫 시도 기준 단계별 비용."""
    stages: dict[str, StageCost]
    images: int
    jobs: int
    per_image_s: float       # per generated image (prompt + inpaint + validation)

    @property
    def total_s(self) -> float:
        return sum(c.total_s for c in self.stages.values())

    def after(self, stage: str) -> float:
        """stage 뒤에 남은 단계들의 추정 시간."""
        if stage not in STAGE_ORDER:
            return 0.0
        later = STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]
        return sum(self.stages[s].total_s for s in later if s in self.stages)


def make_plan(settings: BackgroundSwapSettings, images: int,
              image_size: tuple[int, int] | None, profile: HardwareProfile | None = None) -> Plan:
    """images: frame stride를 적용한 이미지 수. image_size: (width, height), 모르면 SDXL 기본 크기.
    profile: 이 PC의 속도 배율 (없으면 기준 GPU 값 그대로)."""
    def f(stage: str) -> float:
        return profile.factor(speed_key(settings, stage)) if profile is not None else 1.0

    w, h = image_size or (SDXL_NATIVE_PX, SDXL_NATIVE_PX * 3 // 4)
    jobs = images * settings.num_variants
    refs = sum(min(len(v), MAX_PER_AXIS) for v in active_reference_images(settings).values())
    mask_s = MASK_S * f("masks")
    prompt_s = (PROMPT_S if settings.use_llm else 0.0) * f("prompts")
    check_s = (GDINO_CHECK_S + (VLM_CHECK_S if settings.use_vlm else 0.0)) * f("validate")
    gen_s = inpaint_seconds(w, h, settings) * f("inpaint")
    stages = {
        "references": StageCost(LOAD_S["validate"] + LOAD_S["vlm"], REFERENCE_S * f("references"), refs),
        "masks": StageCost(LOAD_S["masks"], mask_s, images),
        "prompts": StageCost(LOAD_S["prompts"] if settings.use_llm else 0.0, prompt_s, jobs),
        "inpaint": StageCost(LOAD_S["inpaint"] + (FAST_LORA_LOAD_S if settings.fast_mode else 0.0),
                             gen_s, jobs),
        "validate": StageCost(LOAD_S["validate"] + (LOAD_S["vlm"] if settings.use_vlm else 0.0),
                              check_s, jobs),
    }
    return Plan(stages=stages, images=images, jobs=jobs, per_image_s=prompt_s + gen_s + check_s)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "1분 미만"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"약 {minutes}분"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"약 {hours}시간 {minutes}분" if minutes else f"약 {hours}시간"
    days, hours = divmod(hours, 24)
    return f"약 {days}일 {hours}시간" if hours else f"약 {days}일"


def format_finish(seconds: float, now: float | None = None) -> str:
    """완료 예상 시각. 오늘이 아니면 날짜도 붙인다."""
    now = time.time() if now is None else now
    end = time.localtime(now + max(0.0, seconds))
    if time.strftime("%Y%m%d", end) == time.strftime("%Y%m%d", time.localtime(now)):
        return time.strftime("%H:%M", end)
    return time.strftime("%m/%d %H:%M", end)


def describe_plan(plan: Plan, settings: BackgroundSwapSettings, now: float | None = None) -> str:
    """실행 전 예상 문구."""
    if plan.jobs <= 0:
        return "예상 시간: 이미지 폴더를 선택하면 계산합니다"
    total = plan.total_s
    text = (f"Run 예상: {format_duration(total)} (완료 {format_finish(total, now)}) · "
            f"{plan.images:,}장 × 변형 {settings.num_variants} = {plan.jobs:,}장, "
            f"장당 약 {plan.per_image_s:.0f}초")
    if settings.max_attempts > 1:
        text += f" · 재시도 시 최대 ×{settings.max_attempts}"
    return text


@dataclass
class EtaTracker:
    """실행 중 남은 시간. 진행 이벤트(stage, done, total)를 받을 때마다 갱신한다."""
    plan: Plan | None = None
    _stage: str = ""
    _first_t: float | None = None     # when the stage's first item finished (excludes model load time)
    _first_done: int = 0
    _per_item: dict[str, float] = field(default_factory=dict)   # measured seconds per item
    _samples: dict[str, int] = field(default_factory=dict)      # number of items behind that measurement

    def speed_ratios(self, base: Plan, min_samples: int = 2) -> dict[str, float]:
        """단계별 (실측 장당 시간 / 기준 GPU 추정 장당 시간). base는 profile 없이 만든 계획."""
        out = {}
        for stage, per in self._per_item.items():
            cost = base.stages.get(stage)
            if cost is not None and cost.per_item_s > 0 and self._samples.get(stage, 0) >= min_samples:
                out[stage] = per / cost.per_item_s
        return out

    def update(self, stage: str, done: int, total: int, now: float | None = None) -> float | None:
        """남은 시간(초). 아직 계산할 수 없으면 None."""
        now = time.monotonic() if now is None else now
        if stage != self._stage or done < self._first_done:
            self._stage, self._first_t, self._first_done = stage, None, 0
        if done >= 1 and self._first_t is None:
            self._first_t, self._first_done = now, done
        if self._first_t is not None and done > self._first_done:
            self._per_item[stage] = (now - self._first_t) / (done - self._first_done)
            self._samples[stage] = done - self._first_done
        per = self._per_item.get(stage)
        if per is None and self.plan is not None and stage in self.plan.stages:
            per = self.plan.stages[stage].per_item_s
        if per is None:
            return None
        remaining = max(0, total - done) * per
        if self.plan is not None and stage in ("references", "masks", "prompts", "inpaint", "validate"):
            remaining += self._later_stages(stage, total)
        return remaining

    def _later_stages(self, stage: str, total: int) -> float:
        """이번 시도 라운드에서 현재 단계 뒤에 남은 단계 (실측 속도가 있으면 그것을 쓴다)."""
        plan = self.plan
        # prompts/inpaint/validate share the job list; before masks, use the planned job count
        jobs = plan.jobs if stage in ("references", "masks") else total
        out = 0.0
        for s in STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]:
            cost = plan.stages.get(s)
            count = cost.count if s == "masks" else jobs if cost is not None else 0
            if cost is None or not count:
                continue
            out += cost.load_s + self._per_item.get(s, cost.per_item_s) * count
        return out
