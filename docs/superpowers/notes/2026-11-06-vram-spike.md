# VRAM spike (2026-11-06) - Background Swap on 8GB

Hardware: RTX 4060 Laptop, 8188 MiB VRAM (15 MiB used at idle), 62GB RAM.
Env: `augmentor/.venv` (`--system-site-packages`), torch 2.13.0+cu130 (system), bitsandbytes 0.50.2.

## Versions (use these APIs in later tasks)

- transformers **5.22.1**
- diffusers **0.42.1**
- huggingface_hub 2.4.0 (httpx2-based). Needs `brotli>=1.2`; the system brotli 1.1.0 makes every download fail with
  `TypeError: process() takes no keyword arguments` (added to `requirements-bg.txt`).
- transformers 5: `from_pretrained(..., dtype=torch.float16)` (the `torch_dtype=` name is the old one; diffusers 0.42 still accepts `torch_dtype=`).
- Grounding DINO `post_process_grounded_object_detection` signature in this version:
  `(outputs, input_ids=None, threshold=0.25, text_threshold=0.25, target_sizes=None, text_labels=None)`
  -> the box-threshold argument is named **`threshold`** (NOT `box_threshold`). Result dict keys: `scores, boxes, text_labels, labels`
  (`labels` is the text label here; use `text_labels`). Call that worked:
  `gp.post_process_grounded_object_detection(go, gi["input_ids"], threshold=0.2, text_threshold=0.2, target_sizes=[(H, W)])[0]`
- SAM2 (`Sam2Model` / `Sam2Processor`, `facebook/sam2.1-hiera-small`), working box-prompt flow:
  ```python
  si = sp(images=img, input_boxes=[boxes], return_tensors="pt").to("cuda")   # boxes: list of [x1,y1,x2,y2] for one image
  with torch.no_grad(): so = sm(**si, multimask_output=False)                 # so.pred_masks shape [1, n_boxes, 1, 256, 256]
  masks = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0] # bool tensor [n_boxes, 1, H, W]
  ```

## Results (synthetic images, see caveat)

All runs: one process per measurement, `reset_peak_memory_stats()` after load, so "peak" = weights resident + one generation.
`alloc` = `max_memory_allocated`, `reserved` = `max_memory_reserved` (the number that matters for real OOM; with cpu offload they differ little).
Time = generation only (30 steps for diffusion), warm disk cache. First-ever SDXL load from disk took 146s, later loads ~2.5s.

| Model | Config | Image (WxH) | Peak alloc GB | Peak reserved GB | Time s | Result |
|---|---|---|---|---|---|---|
| SDXL inpaint (diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | fp16, cpu offload, vae slicing+tiling, 30 st | 512x288 (16:9) | 5.12 | 5.36 | 7.2 | OK |
| SDXL | same | 768x432 (16:9) | 5.17 | 5.42 | 10.7 | OK |
| SDXL | same | 768x576 (4:3) | 5.19 | 5.48 | 12.7 | OK |
| SDXL | same | 1024x576 (16:9) | 5.23 | 5.59 | 15.5 | OK |
| SDXL | same | 1024x768 (4:3) | 5.29 | 5.64 | 19.7 | OK |
| SDXL | same | 1280x720 (16:9) | 5.33 | 5.69 | 23.7 | OK |
| SDXL (extra, beyond brief grid) | same | 1536x864 (16:9) | 5.45 | 5.96 | 32.0 | OK |
| SDXL (extra) | same | 1920x1080 (16:9) | 5.68 | 6.36 | 50.9 | OK, close to 60s limit |
| SD1.5 inpaint (stable-diffusion-v1-5/stable-diffusion-inpainting) | fp16, cpu offload, 30 st | 512x288 | 2.43 | 2.64 | 4.1 | OK |
| SD1.5 | same | 512x384 | 2.45 | 2.66 | 4.5 | OK |
| SD1.5 | same | 768x432 | 2.51 | 2.71 | 6.5 | OK |
| SD1.5 | same | 768x576 | 2.57 | 2.68 | 8.0 | OK |
| Qwen2.5-3B-Instruct | fp16, 128 new tokens | - | 5.76 | 5.83 | 2.4 | OK |
| Qwen2.5-3B-Instruct | 4bit nf4 | - | 2.07 | 2.11 | 1.7 | OK |
| Qwen2.5-VL-3B-Instruct | fp16, 1 image (1280x720, max_pixels=1280*28*28, 1234 input tokens) | 1280x720 | 7.24 | 7.34 | 1.2 | OK but exceeds 7.0 |
| Qwen2.5-VL-3B-Instruct | 4bit nf4 | same | 2.65 | 2.86 | 1.3 | OK |
| SAM2.1-small + GDINO-tiny (both resident) | fp32, 1 image | 1920x1080 | 2.05 | 2.47 | 1.5 | OK |

Peak SDXL VRAM is nearly flat in resolution (cpu offload keeps only the UNet resident, about 5GB); time scales with pixels, so time is the binding constraint, not VRAM.
Qwen sample outputs were sensible (LLM wrote a one-sentence street scene prompt; VLM answered "Yes." for the red-rectangle check in both precisions).

## Decisions

Rule from the brief: `MAX_RESOLUTION_PX_DEFAULT` = largest SDXL long side (grid 512/768/1024/1280) with peak < 7.0GB and time < 60s. All four pass, so 1280. SDXL stays the default inpainter (no need to fall back to SD1.5).
LLM/VLM: fp16 if peak < 7.0GB else 4bit. Qwen2.5-3B fp16 peaks at 5.76GB -> fp16. Qwen2.5-VL-3B fp16 peaks at 7.24GB (7.34 reserved) -> 4bit.

```
MAX_RESOLUTION_PX_DEFAULT = 1280
LLM_LOAD = fp16
VLM_LOAD = 4bit
```

## Caveats / things later tasks should know

- **The real dataset image-size distribution was NOT measured** (no dataset path was available; Step 3 of the brief was replaced by synthetic 4:3 and 16:9 images). It still needs to be confirmed against the real train images. 1920x1080 inputs resized to 1280 long side are within budget; anything larger than the default will be downscaled before inpainting.
- Models are not co-resident in these numbers. Each row is one model alone on the GPU, with an otherwise idle GPU (no desktop/Tk usage). The pipeline must load/unload per stage (LLM -> SAM2/GDINO -> SDXL -> VLM), freeing with `del; gc.collect(); torch.cuda.empty_cache()`. LLM fp16 (5.8GB) + anything else would not fit, so do not keep stages resident together.
- SDXL at 1920 long side took 50.9s, inside the 60s limit with little margin; 1280 gives ~24s. Timings are on a laptop GPU, which may throttle.
- `enable_model_cpu_offload()` needs `accelerate` (installed); the fp16 variant of the SDXL inpainting checkpoint exists, so no fallback to non-variant was needed.
- Model downloads: all default IDs were public, no auth needed. About 20GB of HF cache was used (some of it pre-existing, unrelated models).
