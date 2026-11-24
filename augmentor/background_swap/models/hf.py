"""실제 Hugging Face 모델 래퍼 (Background Swap에서 torch를 import하는 유일한 모듈).

- 래퍼: HFSegmenter(SAM2), HFTextDetector(Grounding DINO), HFPromptGenerator(Qwen2.5 LLM, 한→영 번역 겸용),
  HFInpainter(SDXL/SD 인페인팅), HFValidator(Qwen2.5-VL)
- HFModelFactory: 단계마다 한 모델 계열만 로드하고, 컨텍스트 종료 시(예외 포함) GPU 메모리를 해제
- preflight: 모델을 로드하지 않고 CUDA/모델 접근 가능 여부 점검 (한국어 메시지)
- oom_retry: CUDA OOM 시 캐시 비우고 1회 재시도, 재실패 시 GpuOutOfMemory

정밀도/API 근거: docs/superpowers/notes/2026-11-06-vram-spike.md
(transformers 5.22 / diffusers 0.42, LLM fp16, VLM 4bit nf4, 8GB VRAM)
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForZeroShotObjectDetection,
                          AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration, Sam2Model, Sam2Processor)

from background_swap.common.domain import Detection, GpuOutOfMemory
from background_swap.common.hardware import load_plan
from background_swap.common.settings import BackgroundSwapSettings, is_sdxl
from background_swap.stages.masks import Segmenter, TextDetector
from background_swap.stages.prompts import PromptRequest, build_llm_messages
from background_swap.stages.translate import build_translation_messages

T = TypeVar("T")

DEVICE = "cuda"

# Default load modes (8GB baseline). At runtime HFModelFactory picks them from this PC's VRAM
# (background_swap.common.hardware.load_plan).
LLM_LOAD = "fp16"
VLM_LOAD = "4bit"
INPAINT_OFFLOAD = "model"
VLM_MAX_PIXELS = 1280 * 28 * 28

# Generation parameters (spec Stage 3 / Stage 2 / Stage 4)
INPAINT_STEPS = 30
INPAINT_STRENGTH = 0.99
INPAINT_GUIDANCE = 7.0


@dataclass(frozen=True)
class InpaintRecipe:
    steps: int
    strength: float
    guidance: float
    lora: tuple[str, str] | None = None    # (Hugging Face repo, filename)
    scheduler: str | None = None           # "lcm" | "euler_trailing" | "ddim_trailing"


NORMAL_RECIPE = InpaintRecipe(INPAINT_STEPS, INPAINT_STRENGTH, INPAINT_GUIDANCE)
# Fast mode: Hyper-SD 8-step LoRA (keeps CFG, so negative prompts still work).
# 2026-11-15 sample3 comparison (RTX 4060 Laptop, 1024x768, per image): 30-step default 18.6s, Hyper-SD 6.9s,
# SDXL-Lightning 8 steps 4.9s (distorted, no negative prompt), LCM-LoRA 6 steps 6.0s (blurry).
FAST_RECIPE = InpaintRecipe(8, 1.0, 5.0, ("ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors"),
                            "ddim_trailing")
LLM_TEMPERATURE = 0.9
LLM_TOP_P = 0.95
LLM_MAX_NEW_TOKENS = 96
TRANSLATE_MAX_NEW_TOKENS = 32
VLM_MAX_NEW_TOKENS = 128
DESCRIBE_MAX_NEW_TOKENS = 80


def oom_retry(fn: Callable[[], T]) -> T:
    """CUDA OOM이면 캐시를 비우고 1회 재시도. 재실패 시 GpuOutOfMemory."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        pass  # clean up outside the except block so the traceback releases its tensors
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return fn()
    except torch.cuda.OutOfMemoryError as e:
        msg = str(e)
    gc.collect()
    torch.cuda.empty_cache()
    raise GpuOutOfMemory(msg)


def release(*objs) -> None:
    """래퍼가 잡고 있는 모델 참조를 끊고 GPU 캐시를 비운다 (None은 무시)."""
    for obj in objs:
        close = getattr(obj, "close", None)
        if callable(close):
            close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def has_4bit() -> bool:
    """bitsandbytes가 설치돼 있으면 True (Small 설치에는 없다)."""
    import importlib.util
    return importlib.util.find_spec("bitsandbytes") is not None


def gpu_vram_gb() -> float:
    """GPU 0의 전체 메모리(GiB). CUDA가 없으면 0."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    except Exception:
        pass
    return 0.0


def vlm_compute_dtype() -> torch.dtype:
    """VLM 연산 dtype. Qwen2.5-VL은 fp16 연산에서 활성값이 넘쳐 '!!!!…'(토큰 0 반복)만 내는
    경우가 실측됨(4bit nf4, 합성 이미지 3장 중 1장) → bf16 지원 GPU(RTX 40xx 등)는 bf16, 메모리 동일."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _bnb_4bit(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=compute_dtype)


def _load_kwargs(mode: str, dtype: torch.dtype = torch.float16) -> dict:
    if mode == "4bit":
        return {"quantization_config": _bnb_4bit(dtype), "dtype": dtype, "device_map": DEVICE}
    return {"dtype": dtype, "device_map": DEVICE}


def _to_bool_hw(mask, h: int, w: int) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    while m.ndim > 2:
        m = m[0]
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


class _Closable:
    _attrs: tuple[str, ...] = ()

    def close(self) -> None:
        for name in self._attrs:
            if hasattr(self, name):
                setattr(self, name, None)


def _clear_sam2_position_cache() -> None:
    """transformers 5.22 SAM2는 위치 임베딩을 모듈 수준 lru_cache(maxsize=1)에 GPU 텐서(64MB)로
    남긴다. 모델을 해제해도 남으므로 best-effort로 비운다 (내부 구조가 바뀌면 조용히 무시)."""
    try:
        from transformers.models.sam2 import modeling_sam2
        fn = modeling_sam2.Sam2SinePositionEmbedding.build_sine_position_embedding
        for cell in getattr(fn, "__closure__", None) or ():
            clear = getattr(cell.cell_contents, "cache_clear", None)
            if callable(clear):
                clear()
    except Exception:
        pass


class HFSegmenter(_Closable):
    _attrs = ("model", "processor")

    def close(self) -> None:
        super().close()
        _clear_sam2_position_cache()

    def __init__(self, model_id: str, device: str = DEVICE) -> None:
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def segment_boxes(self, image_rgb: np.ndarray,
                      boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        if not boxes:
            return []
        h, w = image_rgb.shape[:2]
        img = Image.fromarray(image_rgb)
        xyxy = [[float(v) for v in b] for b in boxes]

        def run():
            si = self.processor(images=img, input_boxes=[xyxy], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                so = self.model(**si, multimask_output=False)
            # [n_boxes, 1, H, W] bool
            return self.processor.post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0]

        masks = oom_retry(run)
        return [_to_bool_hw(m.numpy(), h, w) for m in masks]


def match_phrase(label: str, phrases: list[str]) -> str:
    """검출 라벨 문자열 → 가장 잘 맞는 phrase (대소문자 무시 완전 일치 > 부분 문자열 > 첫 phrase)."""
    lab = label.strip().strip(".").strip().lower()
    for p in phrases:
        if p.strip().lower() == lab:
            return p
    if lab:
        for p in phrases:
            pl = p.strip().lower()
            if pl and (lab in pl or pl in lab):
                return p
    return phrases[0]


class HFTextDetector(_Closable):
    _attrs = ("model", "processor")

    def __init__(self, model_id: str, device: str = DEVICE) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image_rgb: np.ndarray, phrases: list[str],
               box_threshold: float, text_threshold: float) -> list[Detection]:
        phrases = [p for p in phrases if p.strip()]
        if not phrases:
            return []
        h, w = image_rgb.shape[:2]
        img = Image.fromarray(image_rgb)
        text = ". ".join(p.strip().lower() for p in phrases) + "."

        def run():
            gi = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                go = self.model(**gi)
            # transformers 5.22: the box threshold argument is `threshold`, not `box_threshold`
            res = self.processor.post_process_grounded_object_detection(
                go, gi["input_ids"], threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=[(h, w)])[0]
            return (res["scores"].float().cpu().tolist(), res["boxes"].float().cpu().tolist(),
                    list(res["text_labels"]))

        scores, boxes, labels = oom_retry(run)
        out: list[Detection] = []
        for score, box, label in zip(scores, boxes, labels):
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, x2 = min(max(x1, 0.0), float(w)), min(max(x2, 0.0), float(w))
            y1, y2 = min(max(y1, 0.0), float(h)), min(max(y2, 0.0), float(h))
            out.append(Detection(box=(x1, y1, x2, y2), score=float(score),
                                 label=match_phrase(str(label), phrases)))
        return out


class HFPromptGenerator(_Closable):
    _attrs = ("model", "tokenizer")

    def __init__(self, model_id: str, class_names: list[str], load: str = LLM_LOAD,
                 device: str = DEVICE) -> None:
        self.device = device
        self.class_names = list(class_names)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **_load_kwargs(load)).eval()

    def generate_batch(self, requests: list[PromptRequest]) -> list[str]:
        # one request at a time instead of batching, so each seed stays deterministic
        return [oom_retry(lambda r=req: self._one(r)) for req in requests]

    def _one(self, req: PromptRequest) -> str:
        tok = self.tokenizer
        text = tok.apply_chat_template(build_llm_messages(req, self.class_names),
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(req.seed)
        with torch.inference_mode():
            out = self.model.generate(
                **enc, do_sample=True, temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def translate_batch(self, texts: list[str]) -> list[str]:
        """한국어 문구를 영어로. 같은 입력이면 같은 출력이 나오도록 탐욕적 디코딩을 쓴다."""
        return [oom_retry(lambda t=text: self._translate(t)) for text in texts]

    def _translate(self, text: str) -> str:
        tok = self.tokenizer
        prompt = tok.apply_chat_template(build_translation_messages(text), tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=TRANSLATE_MAX_NEW_TOKENS,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()


def _apply_fast_lora(pipe, recipe: InpaintRecipe) -> None:
    """가속 LoRA를 UNet에 합치고(fuse) 그에 맞는 스케줄러로 바꾼다. 합친 뒤 LoRA 층은 내려 메모리를 아낀다."""
    from diffusers import DDIMScheduler, EulerDiscreteScheduler, LCMScheduler
    from huggingface_hub import hf_hub_download

    repo, filename = recipe.lora
    pipe.load_lora_weights(hf_hub_download(repo, filename))
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    config = pipe.scheduler.config
    if recipe.scheduler == "lcm":
        pipe.scheduler = LCMScheduler.from_config(config)
    elif recipe.scheduler == "euler_trailing":
        pipe.scheduler = EulerDiscreteScheduler.from_config(config, timestep_spacing="trailing")
    elif recipe.scheduler == "ddim_trailing":
        pipe.scheduler = DDIMScheduler.from_config(config, timestep_spacing="trailing")


class HFInpainter(_Closable):
    _attrs = ("pipe",)

    def __init__(self, model_id: str, offload: str = INPAINT_OFFLOAD, fast: bool = False,
                 device: str = DEVICE) -> None:
        self.device = device
        self.recipe = FAST_RECIPE if fast else NORMAL_RECIPE
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                model_id, dtype=torch.float16, variant="fp16")
        except (OSError, ValueError):
            # checkpoint without an fp16 variant
            pipe = AutoPipelineForInpainting.from_pretrained(model_id, dtype=torch.float16)
        if self.recipe.lora is not None:
            _apply_fast_lora(pipe, self.recipe)
        if offload == "none":
            pipe.to(device)
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.native_px = 1024 if "XL" in type(pipe).__name__ else 512

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str,
                negative_prompt: str, seed: int) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        image = Image.fromarray(np.ascontiguousarray(image_rgb[..., :3]))
        mask_img = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255)  # L mode, 255 = repaint

        def run():
            # with cpu offload the execution device is still cuda, so a cuda generator works (verified)
            gen = torch.Generator(self.device).manual_seed(int(seed))
            result = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=image,
                               mask_image=mask_img, width=w, height=h,
                               num_inference_steps=self.recipe.steps, strength=self.recipe.strength,
                               guidance_scale=self.recipe.guidance, generator=gen)
            return result.images[0]

        out = np.asarray(oom_retry(run).convert("RGB"), dtype=np.uint8)
        if out.shape[:2] != (h, w):
            interp = cv2.INTER_AREA if out.shape[0] * out.shape[1] > h * w else cv2.INTER_CUBIC
            out = cv2.resize(out, (w, h), interpolation=interp)
        return np.ascontiguousarray(out, dtype=np.uint8)


class HFValidator(_Closable):
    _attrs = ("model", "processor")

    def __init__(self, model_id: str, load: str = VLM_LOAD, device: str = DEVICE) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Same as the spike's max_pixels=1280*28*28. from_pretrained(max_pixels=) works in transformers 5.22
        # but warns of removal in 5.23, and size= is ignored, so set it after loading.
        ip = self.processor.image_processor
        shortest = (ip.size.get("shortest_edge") if isinstance(ip.size, dict)
                    else getattr(ip.size, "shortest_edge", None)) or 56 * 56
        ip.size = {"longest_edge": VLM_MAX_PIXELS, "shortest_edge": shortest}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **_load_kwargs(load, vlm_compute_dtype())).eval()

    def judge_batch(self, images_rgb: list[np.ndarray], prompt: str) -> list[str]:
        # one image per conversation, sequentially (more stable than padded batches, lower peak memory)
        return [oom_retry(lambda im=img: self._one(im, prompt)) for img in images_rgb]

    def describe(self, image_rgb: np.ndarray, prompt: str) -> str:
        """참고 사진 설명 (판정보다 긴 문장이 필요해 토큰 한도를 따로 둔다)."""
        return oom_retry(lambda: self._one(image_rgb, prompt, DESCRIBE_MAX_NEW_TOKENS))

    def _one(self, image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 0) -> str:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[Image.fromarray(image_rgb)],
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=max_new_tokens or VLM_MAX_NEW_TOKENS)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _load(cls: Callable[..., T], *args) -> T:
    """모델 로드도 OOM 재시도 대상 (다른 앱이 VRAM을 쥐고 있으면 로드가 가장 흔한 OOM 지점).

    첫 시도에서 반쯤 올라간 가중치는 생성 중이던 객체와 함께 참조가 끊기므로,
    oom_retry가 except 블록 밖에서 gc.collect()/empty_cache()를 호출할 때 해제된 뒤 재시도한다.
    """
    return oom_retry(lambda: cls(*args))


class HFModelFactory:
    """ModelFactory 구현. 각 컨텍스트는 진입 시 로드, 종료 시(예외 포함) 해제."""

    def __init__(self, settings: BackgroundSwapSettings, class_names: list[str]) -> None:
        self.settings = settings
        self.class_names = list(class_names)
        self.load_plan = load_plan(gpu_vram_gb(), has_4bit())

    @property
    def ids(self):
        return self.settings.models

    @contextmanager
    def mask_models(self) -> Iterator[tuple[Segmenter, TextDetector]]:
        seg = det = None
        try:
            seg = _load(HFSegmenter, self.ids.segmenter)
            det = _load(HFTextDetector, self.ids.detector)
            yield seg, det
        finally:
            release(seg, det)
            seg = det = None

    @contextmanager
    def translator(self) -> Iterator[HFPromptGenerator]:
        """한→영 번역용 LLM (장면 프롬프트 생성과 같은 모델)."""
        with self.prompt_model() as gen:
            yield gen

    @contextmanager
    def prompt_model(self) -> Iterator[HFPromptGenerator]:
        gen = None
        try:
            gen = _load(HFPromptGenerator, self.ids.llm, self.class_names, self.load_plan.llm_load)
            yield gen
        finally:
            release(gen)
            gen = None

    @contextmanager
    def inpainter(self) -> Iterator[HFInpainter]:
        inp = None
        try:
            inp = _load(HFInpainter, self.ids.inpainter, self.load_plan.inpaint_offload,
                        self.settings.fast_mode)
            yield inp
        finally:
            release(inp)
            inp = None

    @contextmanager
    def detector(self) -> Iterator[HFTextDetector]:
        det = None
        try:
            det = _load(HFTextDetector, self.ids.detector)
            yield det
        finally:
            release(det)
            det = None

    @contextmanager
    def describer(self) -> Iterator[HFValidator]:
        """참고 사진 설명용 VLM (검수와 같은 모델)."""
        with self.validator() as val:
            yield val

    @contextmanager
    def validator(self) -> Iterator[HFValidator]:
        val = None
        try:
            val = _load(HFValidator, self.ids.vlm, self.load_plan.vlm_load)
            yield val
        finally:
            release(val)
            val = None


_MODEL_LABELS = (
    ("segmenter", "세그멘터(SAM2)"),
    ("detector", "검출기(Grounding DINO)"),
    ("llm", "LLM"),
    ("inpainter", "인페인팅 모델"),
    ("vlm", "VLM"),
)


def preflight(settings: BackgroundSwapSettings) -> list[str]:
    """실행 전 점검. 문제가 없으면 빈 목록. 모델은 로드하지 않는다."""
    from huggingface_hub import model_info, try_to_load_from_cache

    problems: list[str] = []
    if not torch.cuda.is_available():
        problems.append("CUDA GPU를 사용할 수 없습니다. NVIDIA 드라이버와 CUDA용 torch 설치를 확인하세요.")
    checks = [(label, getattr(settings.models, field),
               "model_index.json" if field == "inpainter" else "config.json")
              for field, label in _MODEL_LABELS
              if not ((field == "llm" and not settings.use_llm) or (field == "vlm" and not settings.use_vlm))]
    if settings.fast_mode and FAST_RECIPE.lora is not None:
        checks.append(("빠른 모드 LoRA", *FAST_RECIPE.lora))
        if not is_sdxl(settings.models.inpainter):
            problems.append("빠른 모드는 SDXL 인페인팅 모델(Large)에서만 쓸 수 있습니다. "
                            "Small 모델은 이미 빠르니 빠른 모드를 끄세요.")
        import importlib.util
        if importlib.util.find_spec("peft") is None:
            problems.append("빠른 모드에는 peft 패키지가 필요합니다: make venv (Large 설치)")
    for label, model_id, filename in checks:
        try:
            cached = try_to_load_from_cache(model_id, filename)
        except Exception:
            cached = None
        if isinstance(cached, str):
            continue
        try:
            model_info(model_id)
        except Exception as e:
            problems.append(f"{label} 모델 '{model_id}'에 접근할 수 없습니다 "
                            f"(로컬 캐시에 없고 Hugging Face 확인 실패): {type(e).__name__}: {e}")
    return problems
