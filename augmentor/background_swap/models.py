"""실제 Hugging Face 모델 래퍼 (Background Swap에서 torch를 import하는 유일한 모듈).

- 래퍼: HFSegmenter(SAM2), HFTextDetector(Grounding DINO), HFPromptGenerator(Qwen2.5 LLM),
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
from typing import Callable, Iterator, TypeVar

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForZeroShotObjectDetection,
                          AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration, Sam2Model, Sam2Processor)

from .domain import Detection, GpuOutOfMemory
from .masks import Segmenter, TextDetector
from .prompts import PromptRequest, build_llm_messages
from .settings import BackgroundSwapSettings

T = TypeVar("T")

DEVICE = "cuda"

# VRAM spike 결정 (LLM fp16 5.8GB, VLM fp16 7.2GB > 7.0 → 4bit)
LLM_LOAD = "fp16"
VLM_LOAD = "4bit"
VLM_MAX_PIXELS = 1280 * 28 * 28

# 생성 파라미터 (spec Stage 3 / Stage 2 / Stage 4)
INPAINT_STEPS = 30
INPAINT_STRENGTH = 0.99
INPAINT_GUIDANCE = 7.0
LLM_TEMPERATURE = 0.9
LLM_TOP_P = 0.95
LLM_MAX_NEW_TOKENS = 96
VLM_MAX_NEW_TOKENS = 128


# ── 공용 헬퍼 ──────────────────────────────────────────────────────

def oom_retry(fn: Callable[[], T]) -> T:
    """CUDA OOM이면 캐시를 비우고 1회 재시도. 재실패 시 GpuOutOfMemory."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        pass  # except 블록 밖에서 정리해야 traceback이 잡고 있던 텐서가 풀린다
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


# ── SAM2 ───────────────────────────────────────────────────────────

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


# ── Grounding DINO ─────────────────────────────────────────────────

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
            # transformers 5.22: 박스 임계값 인자명은 threshold (box_threshold 아님)
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


# ── LLM 프롬프트 생성 ──────────────────────────────────────────────

class HFPromptGenerator(_Closable):
    _attrs = ("model", "tokenizer")

    def __init__(self, model_id: str, class_names: list[str], load: str = LLM_LOAD,
                 device: str = DEVICE) -> None:
        self.device = device
        self.class_names = list(class_names)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **_load_kwargs(load)).eval()

    def generate_batch(self, requests: list[PromptRequest]) -> list[str]:
        # 배치 대신 요청별 순차 생성: 시드별 결정성 우선
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


# ── 인페인팅 ───────────────────────────────────────────────────────

class HFInpainter(_Closable):
    _attrs = ("pipe",)

    def __init__(self, model_id: str, device: str = DEVICE) -> None:
        self.device = device
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                model_id, dtype=torch.float16, variant="fp16")
        except (OSError, ValueError):
            # fp16 variant가 없는 체크포인트
            pipe = AutoPipelineForInpainting.from_pretrained(model_id, dtype=torch.float16)
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
        mask_img = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255)  # L, 255=다시 그림

        def run():
            # cpu offload에서도 실행 장치는 cuda이므로 cuda 생성기가 동작한다 (실측)
            gen = torch.Generator(self.device).manual_seed(int(seed))
            result = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=image,
                               mask_image=mask_img, width=w, height=h,
                               num_inference_steps=INPAINT_STEPS, strength=INPAINT_STRENGTH,
                               guidance_scale=INPAINT_GUIDANCE, generator=gen)
            return result.images[0]

        out = np.asarray(oom_retry(run).convert("RGB"), dtype=np.uint8)
        if out.shape[:2] != (h, w):
            interp = cv2.INTER_AREA if out.shape[0] * out.shape[1] > h * w else cv2.INTER_CUBIC
            out = cv2.resize(out, (w, h), interpolation=interp)
        return np.ascontiguousarray(out, dtype=np.uint8)


# ── VLM 검증 ───────────────────────────────────────────────────────

class HFValidator(_Closable):
    _attrs = ("model", "processor")

    def __init__(self, model_id: str, load: str = VLM_LOAD, device: str = DEVICE) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        # spike의 max_pixels=1280*28*28과 동일. transformers 5.22에서 from_pretrained(max_pixels=)는
        # 동작하지만 5.23 제거 예고 경고가 나고, size= 인자는 무시되므로 로드 후 직접 설정한다.
        ip = self.processor.image_processor
        shortest = (ip.size.get("shortest_edge") if isinstance(ip.size, dict)
                    else getattr(ip.size, "shortest_edge", None)) or 56 * 56
        ip.size = {"longest_edge": VLM_MAX_PIXELS, "shortest_edge": shortest}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **_load_kwargs(load, vlm_compute_dtype())).eval()

    def judge_batch(self, images_rgb: list[np.ndarray], prompt: str) -> list[str]:
        # 이미지 1장 = 대화 1개, 순차 실행 (패딩 배치보다 결과가 안정적이고 피크 메모리가 낮음)
        return [oom_retry(lambda im=img: self._one(im, prompt)) for img in images_rgb]

    def _one(self, image_rgb: np.ndarray, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[Image.fromarray(image_rgb)],
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=VLM_MAX_NEW_TOKENS)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ── 팩토리 ─────────────────────────────────────────────────────────

class HFModelFactory:
    """ModelFactory 구현. 각 컨텍스트는 진입 시 로드, 종료 시(예외 포함) 해제."""

    def __init__(self, settings: BackgroundSwapSettings, class_names: list[str]) -> None:
        self.settings = settings
        self.class_names = list(class_names)

    @property
    def ids(self):
        return self.settings.models

    @contextmanager
    def mask_models(self) -> Iterator[tuple[Segmenter, TextDetector]]:
        seg = det = None
        try:
            seg = HFSegmenter(self.ids.segmenter)
            det = HFTextDetector(self.ids.detector)
            yield seg, det
        finally:
            release(seg, det)
            seg = det = None

    @contextmanager
    def prompt_model(self) -> Iterator[HFPromptGenerator]:
        gen = None
        try:
            gen = HFPromptGenerator(self.ids.llm, self.class_names)
            yield gen
        finally:
            release(gen)
            gen = None

    @contextmanager
    def inpainter(self) -> Iterator[HFInpainter]:
        inp = None
        try:
            inp = HFInpainter(self.ids.inpainter)
            yield inp
        finally:
            release(inp)
            inp = None

    @contextmanager
    def detector(self) -> Iterator[HFTextDetector]:
        det = None
        try:
            det = HFTextDetector(self.ids.detector)
            yield det
        finally:
            release(det)
            det = None

    @contextmanager
    def validator(self) -> Iterator[HFValidator]:
        val = None
        try:
            val = HFValidator(self.ids.vlm)
            yield val
        finally:
            release(val)
            val = None


# ── 사전 점검 ──────────────────────────────────────────────────────

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
    for field, label in _MODEL_LABELS:
        model_id = getattr(settings.models, field)
        filename = "model_index.json" if field == "inpainter" else "config.json"
        try:
            cached = try_to_load_from_cache(model_id, filename)
        except Exception:  # 잘못된 repo id 형식 등
            cached = None
        if isinstance(cached, str):
            continue
        try:
            model_info(model_id)
        except Exception as e:
            problems.append(f"{label} 모델 '{model_id}'에 접근할 수 없습니다 "
                            f"(로컬 캐시에 없고 Hugging Face 확인 실패): {type(e).__name__}: {e}")
    return problems
