# Background Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 라벨링된 YOLO 이미지에서 객체와 지정한 유지 배경은 원본 픽셀 그대로 두고, 나머지 배경만 LLM 프롬프트 + 인페인팅으로 교체해 검증된 증강 이미지를 만드는 GUI 통합 배치 파이프라인을 만든다.

**Architecture:** `augmentor/background_swap/` 패키지가 단계별 일괄 처리(입력 → 마스크 → 프롬프트 → 인페인팅·합성 → 검증 → 리포트 → export)를 수행하고, 모든 모델은 Protocol 뒤에 숨겨 가짜 구현으로 테스트한다. torch를 쓰는 코드는 `models.py`에만 있고, GUI 탭(`gui/background_tab.py`)은 설정을 만들어 워커 서브프로세스(`python -m background_swap`)를 실행하고 stdout JSON 진행 라인을 읽는다.

**Tech Stack:** Python 3, Tkinter/ttk, OpenCV, NumPy, Pillow, PyYAML, pytest / (워커 전용) torch, transformers(SAM2, Grounding DINO, Qwen2.5, Qwen2.5-VL), diffusers(AutoPipelineForInpainting), accelerate, bitsandbytes, huggingface_hub

**Spec:** `docs/superpowers/specs/2026-11-06-background-swap-design.md`

## Global Constraints

- 모든 명령은 `dataset_helper/augmentor/`에서 실행한다. import 루트는 `augmentor/` (기존 `from core.label import BBox` 방식 유지).
- 테스트: `python3 -m pytest` (기존 66개 테스트가 계속 통과해야 함). GPU 테스트는 `@pytest.mark.gpu`, 기본 실행에서 제외.
- **torch / transformers / diffusers / bitsandbytes는 `background_swap/models.py`에서만 import한다.** GUI와 나머지 모듈은 torch 없이 import 가능해야 한다.
- 무거운 의존성은 `augmentor/requirements-bg.txt`, 기본 `requirements.txt`에는 `pyyaml`만 추가한다.
- 파일 이름: 생성 `generated/{stem}_bgswap{n}_a{k}.png`, 수락 `Accepted/images/{stem}_bgswap{n}.jpg`, 거부 `Rejected/{stem}_bgswap{n}_a{k}.jpg`. `n`(variant)과 `k`(attempt)는 1부터 시작.
- 출력 이미지는 항상 원본 해상도, 라벨은 원본 `.txt`를 그대로 복사한다.
- `object_protect ∪ keep` 내부 픽셀은 출력에서 원본과 바이트 단위로 동일해야 한다.
- 모든 산출물 파일은 임시 파일에 쓴 뒤 `os.replace`로 교체(원자적 쓰기).
- 기본값(스펙 §5): `num_variants=3`, `max_attempts=3`, `seed=0`, `frame_stride=1`, `small_box_px=32`, `dilate_px=6`, `shadow_extend_ratio=0.15`, `feather_px=4`, `min_inpaint_ratio=0.15`, `min_change=20.0`, `resolution_mode="auto"`, `manual_resolution_px=1024`, KeepItem `h_tol=10, s_tol=60, v_tol=60, largest_only=True, convex_hull=True, min_area_ratio=0.05`.
- 검출 임계값: keep text GDINO box 0.35 / text 0.25, 새 객체 검사 score ≥ 0.35 · IoU < 0.3 · 검출 박스 면적의 ≥ 50%가 inpaint 영역, recall 판정 IoU ≥ 0.5.
- 인페인팅: strength 0.99, steps 30, guidance 7.0, fp16, `enable_model_cpu_offload()`, VAE slicing/tiling.
- 모델 기본 ID: segmenter `facebook/sam2.1-hiera-small`, detector `IDEA-Research/grounding-dino-tiny`, llm `Qwen/Qwen2.5-3B-Instruct`, inpainter `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, vlm `Qwen/Qwen2.5-VL-3B-Instruct`.
- UI 문구는 기존 탭처럼 한국어.

## Review Focus

- **빈 라벨 파일 / 라벨 없는 배경 이미지**: 객체 0개인 이미지도 keep·inpaint 마스크만으로 정상 처리되어야 한다 (Task 5 `test_build_mask_set_no_boxes`).
- **RGBA·그레이스케일 PNG 입력**: 3채널 BGR로 읽혀 출력도 3채널 JPG가 되어야 한다 (Task 3 `test_load_image_bgr_converts_rgba_and_gray`).
- **범위를 벗어나거나 크기 0인 bbox** (`x+w/2 > 1`, `w == 0`): 이미지 경계로 클립하고 크기 0 박스는 무시, 예외 없이 진행 (Task 5 `test_object_core_clips_and_ignores_degenerate_boxes`).
- **Stop/크래시로 잘린 jsonl 마지막 줄**: 재개 시 해당 줄만 무시하고 이어서 진행 (Task 2 `test_read_jsonl_ignores_truncated_last_line`).
- **출력 폴더 = images/labels 폴더**: 결과가 원본과 섞이므로 설정 검증에서 거부 (Task 1 `test_validate_settings_rejects_output_equal_to_input`).

---

### Task 0: VRAM 실측 스파이크 + 의존성 파일

버릴 코드로 8GB에서의 실측값을 얻어 `max_resolution_px` 기본값과 LLM/VLM 로딩 정밀도를 확정한다. 측정 스크립트는 커밋하지 않고, 결과 노트와 의존성 파일만 커밋한다.

**Files:**
- Create: `augmentor/requirements-bg.txt`
- Modify: `augmentor/requirements.txt` (`pyyaml` 추가)
- Create: `docs/superpowers/notes/2026-11-06-vram-spike.md`
- Scratch (커밋 안 함): 세션 scratchpad의 `vram_spike.py`

**Interfaces:**
- Produces: 노트에 기록된 `MAX_RESOLUTION_PX_DEFAULT` 값(정수)과 `LLM_LOAD = "fp16" | "4bit"`, `VLM_LOAD = "fp16" | "4bit"` 결정 — Task 1, Task 12가 사용.

- [ ] **Step 1: 의존성 파일 작성**

`requirements-bg.txt`: `transformers>=4.56`, `diffusers>=0.35`, `accelerate`, `bitsandbytes`, `huggingface_hub`, `safetensors` (torch는 이미 설치된 2.13+cu130을 사용 — 파일에 주석으로 "torch는 CUDA 버전에 맞게 별도 설치" 명시). `requirements.txt`에 `pyyaml` 추가.

- [ ] **Step 2: 설치 확인**

Run: `pip install -r requirements.txt -r requirements-bg.txt && python3 -c "import transformers, diffusers, bitsandbytes; print(transformers.__version__, diffusers.__version__)"`
Expected: 버전 두 개 출력, 에러 없음.

- [ ] **Step 3: 데이터셋 이미지 크기 분포 확인**

사용자에게 실제 train images 폴더 경로를 묻고, PIL로 (w, h) 빈도 상위 5개를 출력해 노트에 기록.

- [ ] **Step 4: 스파이크 스크립트로 측정**

각 측정 전 `torch.cuda.reset_peak_memory_stats()`, 후 `torch.cuda.max_memory_allocated()`와 경과 시간 기록. 데이터셋 대표 비율로:
- SDXL inpaint (기본 ID): 긴 변 512 / 768 / 1024 / 1280, 30 steps, cpu offload
- SD1.5 inpaint (`stable-diffusion-v1-5/stable-diffusion-inpainting`): 512 / 768
- Qwen2.5-3B-Instruct: fp16 및 4bit(nf4)로 프롬프트 1개 생성
- Qwen2.5-VL-3B-Instruct: fp16 및 4bit로 이미지 1장 판정
- SAM2-small + GDINO-tiny 동시 로딩 후 이미지 1장

- [ ] **Step 5: 결정 기록**

`MAX_RESOLUTION_PX_DEFAULT` = 피크 VRAM < 7.0GB이고 변형당 시간 < 60s인 가장 큰 SDXL 긴 변 (모두 실패면 SD1.5를 기본 inpainter로 바꾸자는 메모 + 사용자 확인). LLM/VLM은 fp16 피크 < 7.0GB면 fp16, 아니면 4bit. 표와 결정을 노트에 작성.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-bg.txt ../docs/superpowers/notes/2026-11-06-vram-spike.md
git commit -m "chore: add background swap deps and VRAM spike results"
```

---

### Task 1: 설정과 도메인 타입

**Files:**
- Create: `augmentor/background_swap/__init__.py` (빈 파일, torch import 금지)
- Create: `augmentor/background_swap/domain.py`
- Create: `augmentor/background_swap/settings.py`
- Test: `augmentor/tests/test_bg_settings.py`

**Interfaces:**
- Consumes: Task 0의 `MAX_RESOLUTION_PX_DEFAULT`.
- Produces (`domain.py`):
  - `@dataclass(frozen=True) ImageItem(stem: str, image_path: Path, label_path: Path, width: int, height: int)`
  - `@dataclass Detection(box: tuple[float, float, float, float], score: float, label: str)` — box는 픽셀 xyxy
  - `@dataclass MaskSet(object_core: np.ndarray, object_protect: np.ndarray, keep: np.ndarray, inpaint: np.ndarray, warnings: list[str], skip_reason: str | None)` — 마스크는 HxW `bool`; `protect` 프로퍼티 = `object_protect | keep`
  - `@dataclass Job(stem: str, variant: int, attempt: int, seed: int, scene: dict[str, str], prompt: str, negative_prompt: str)` + `to_dict()` / `Job.from_dict(d)`
  - `@dataclass ValidationResult(accepted: bool, reasons: list[str], checks: dict[str, object])`
  - `class GpuOutOfMemory(RuntimeError)`
  - 상태 상수 `ACCEPTED = "accepted"`, `REJECTED = "rejected"`, `SKIPPED = "skipped"`, `EXHAUSTED = "exhausted"`
- Produces (`settings.py`):
  - `@dataclass KeepItem(name: str, method: Literal["color", "text"], required: bool = False, prompt: str = "", hsv_lower: list[int] | None = None, hsv_upper: list[int] | None = None, h_tol: int = 10, s_tol: int = 60, v_tol: int = 60, largest_only: bool = True, convex_hull: bool = True, min_area_ratio: float = 0.05)`
  - `@dataclass ModelIds(segmenter, detector, llm, inpainter, vlm)` — 기본값은 Global Constraints
  - `@dataclass BackgroundSwapSettings` — 필드: `images_dir: Path | None = None`, `labels_dir: Path | None = None`, `output_dir: Path | None = None`, `class_names_file: Path | None = None`, `frame_stride`, `keep_items: list[KeepItem]`, `num_variants`, `max_attempts`, `seed`, `use_llm = True`, `use_vlm = True`, `scene_axes: dict[str, list[str]]` (기본 `DEFAULT_SCENE_AXES`), `resolution_mode`, `max_resolution_px = MAX_RESOLUTION_PX_DEFAULT`, `manual_resolution_px`, `small_box_px`, `dilate_px`, `shadow_extend_ratio`, `feather_px`, `min_inpaint_ratio`, `min_change`, `allow_non_train = False`, `models: ModelIds`
  - `DEFAULT_SCENE_AXES: dict[str, list[str]]` — 스펙 §5 값 그대로 (키 순서: venue, lighting, surroundings_floor, context, camera)
  - `GENERATION_FIELDS: tuple[str, ...]` — 결과에 영향을 주는 필드 (`output_dir`, `allow_non_train`, `class_names_file`를 제외한 전부)
  - `settings_to_dict(s) -> dict`, `settings_from_dict(d) -> BackgroundSwapSettings` (누락 키는 기본값, Path는 문자열 ↔ Path)
  - `save_settings(s, path: Path) -> None` (원자적 쓰기), `load_settings(path: Path) -> BackgroundSwapSettings`
  - `generation_diff(a, b) -> list[str]` — 값이 다른 `GENERATION_FIELDS` 이름
  - `validate_settings(s) -> list[str]` — 한국어 오류 메시지 목록

- [ ] **Step 1: Write the failing tests**

```python
def test_roundtrip_preserves_all_fields(tmp_path):
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"out",
        keep_items=[KeepItem("field", "color", required=True, hsv_lower=[35, 40, 40], hsv_upper=[85, 255, 255]),
                    KeepItem("fence", "text", prompt="wooden arena fence")])
    save_settings(s, tmp_path/"run_settings.json")
    assert load_settings(tmp_path/"run_settings.json") == s

def test_from_dict_fills_missing_with_defaults():
    s = settings_from_dict({"num_variants": 5})
    assert s.num_variants == 5 and s.max_attempts == 3 and s.dilate_px == 6
    assert list(s.scene_axes) == ["venue", "lighting", "surroundings_floor", "context", "camera"]
    assert s.models.inpainter == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

def test_generation_diff_ignores_output_dir(tmp_path):
    a = BackgroundSwapSettings(output_dir=tmp_path/"a")
    b = BackgroundSwapSettings(output_dir=tmp_path/"b", seed=7)
    assert generation_diff(a, b) == ["seed"]

def test_validate_settings_requires_dirs():
    errs = validate_settings(BackgroundSwapSettings())
    assert len(errs) == 3   # images_dir, labels_dir, output_dir 미지정

def test_validate_settings_rejects_output_equal_to_input(tmp_path):
    (tmp_path/"img").mkdir(); (tmp_path/"lbl").mkdir()
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"img")
    assert any("출력" in e for e in validate_settings(s))

def test_validate_settings_color_item_needs_hsv(tmp_path):
    for d in ("img", "lbl"): (tmp_path/d).mkdir()
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"out",
                               keep_items=[KeepItem("field", "color")])
    assert any("field" in e for e in validate_settings(s))

def test_validate_settings_ranges(tmp_path):
    for d in ("img", "lbl"): (tmp_path/d).mkdir()
    s = BackgroundSwapSettings(images_dir=tmp_path/"img", labels_dir=tmp_path/"lbl", output_dir=tmp_path/"out",
                               num_variants=0, max_attempts=11, frame_stride=0)
    assert len(validate_settings(s)) == 3
```

`validate_settings` 규칙: images/labels 존재, output 지정, output ≠ images/labels (resolve 후 비교), `num_variants` 1–20, `max_attempts` 1–10, `frame_stride` 1–100, color 항목은 `hsv_lower/upper` 필수, text 항목은 `prompt` 비어있지 않음, `scene_axes` 각 축 값 1개 이상.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_settings.py -v`
Expected: FAIL (`ModuleNotFoundError: background_swap`)

- [ ] **Step 3: Implement `domain.py`, `settings.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_settings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/ tests/test_bg_settings.py
git commit -m "feat(background_swap): add settings and domain types"
```

---

### Task 2: 원자적 저장소 + 진행 출력

**Files:**
- Create: `augmentor/background_swap/store.py`
- Create: `augmentor/background_swap/progress.py`
- Test: `augmentor/tests/test_bg_store.py`

**Interfaces:**
- Produces (`store.py`):
  - `atomic_write_text(path: Path, text: str) -> None`, `atomic_write_bytes(path, data: bytes) -> None` (부모 폴더 생성, 같은 폴더 임시 파일 → `os.replace`)
  - `imwrite_atomic(path: Path, img: np.ndarray, jpeg_quality: int = 95) -> None` (`cv2.imencode` 확장자 기준 → `atomic_write_bytes`)
  - `write_mask(path, mask: np.ndarray) -> None` (bool → 0/255 PNG), `read_mask(path) -> np.ndarray` (bool)
  - `append_jsonl(path, record: dict) -> None` (한 줄 + `\n`, `flush` + `os.fsync`)
  - `read_jsonl(path) -> list[dict]` (없으면 `[]`, 파싱 실패 줄은 무시)
  - `reset_output_dir(output_dir: Path) -> None` — `items.jsonl, masks/, jobs.jsonl, generated/, Accepted/, Rejected/, Manifest.jsonl, report.json, run_settings.json`만 삭제 (다른 파일은 보존)
- Produces (`progress.py`):
  - `class Progress(Protocol)`: `stage(name: str, total: int) -> None`, `advance(n: int = 1) -> None`, `event(kind: str, **fields) -> None`
  - `JsonLinesProgress(stream=sys.stdout)` — `stage`/`advance`마다 `{"stage": name, "done": d, "total": t}` 한 줄, `event`는 `{"event": kind, ...}` 한 줄, 매번 flush
  - `NullProgress()`

- [ ] **Step 1: Write the failing tests**

```python
def test_read_jsonl_ignores_truncated_last_line(tmp_path):
    p = tmp_path/"m.jsonl"
    append_jsonl(p, {"a": 1}); append_jsonl(p, {"a": 2})
    with open(p, "a") as f: f.write('{"a": 3, "b"')
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]

def test_append_after_truncated_line_is_readable(tmp_path):
    p = tmp_path/"m.jsonl"
    with open(p, "w") as f: f.write('{"a": 1}\n{"trunc')
    append_jsonl(p, {"a": 2})
    assert {"a": 2} in read_jsonl(p)

def test_mask_roundtrip(tmp_path):
    m = np.zeros((4, 5), bool); m[1:3, 2:4] = True
    write_mask(tmp_path/"m.png", m)
    assert (read_mask(tmp_path/"m.png") == m).all()

def test_atomic_write_leaves_no_temp_files(tmp_path):
    atomic_write_text(tmp_path/"x"/"a.txt", "hi")
    assert [p.name for p in (tmp_path/"x").iterdir()] == ["a.txt"]

def test_reset_output_dir_keeps_unrelated(tmp_path):
    (tmp_path/"masks").mkdir(); (tmp_path/"Manifest.jsonl").write_text("")
    (tmp_path/"notes.txt").write_text("keep")
    reset_output_dir(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]

def test_json_lines_progress_format():
    buf = io.StringIO(); p = JsonLinesProgress(buf)
    p.stage("masks", 2); p.advance(); p.event("done")
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert lines == [{"stage": "masks", "done": 0, "total": 2},
                     {"stage": "masks", "done": 1, "total": 2}, {"event": "done"}]
```

`append_jsonl`은 파일이 `\n`으로 끝나지 않으면 먼저 `\n`을 쓴다 (두 번째 테스트).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_store.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `store.py`, `progress.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/store.py background_swap/progress.py tests/test_bg_store.py
git commit -m "feat(background_swap): add atomic store and JSON progress"
```

---

### Task 3: 입력 선택 + 클래스 이름 + `_bgswap` 제외

**Files:**
- Create: `augmentor/background_swap/inputs.py`
- Create: `augmentor/background_swap/classes.py`
- Modify: `augmentor/core/pipeline.py` (`_collect_originals`에 `"_bgswap"` 제외 추가)
- Test: `augmentor/tests/test_bg_inputs.py`, `augmentor/tests/test_pipeline.py` (테스트 1개 추가)

**Interfaces:**
- Consumes: `core.pipeline._collect_originals`, `domain.ImageItem`.
- Produces (`inputs.py`):
  - `NON_TRAIN_PARTS = {"val", "valid", "validation", "test"}`
  - `looks_non_train(path: Path) -> bool` — 경로 구성요소(소문자) 중 하나가 `NON_TRAIN_PARTS`에 정확히 일치
  - `load_image_bgr(path: Path) -> np.ndarray | None` — `cv2.imread(..., cv2.IMREAD_COLOR)` (RGBA/그레이 → 3채널)
  - `collect_items(images_dir: Path, labels_dir: Path, frame_stride: int) -> tuple[list[ImageItem], list[tuple[str, str]]]` — `_collect_originals` 결과에서 `[::frame_stride]` 선택 후, 라벨 없음 `"missing_label"`, 열기 실패 `"unreadable_image"`, 같은 stem 두 번째 이후 `"duplicate_stem"`을 skipped로 반환. 크기는 `PIL.Image.open(p).size`.
- Produces (`classes.py`):
  - `find_class_names_file(images_dir: Path, labels_dir: Path) -> Path | None` — 후보 이름 순서 `classes.txt`, `data.yaml`, `obj.names`; 폴더 순서 labels_dir, images_dir, labels_dir.parent, images_dir.parent; 첫 존재 파일
  - `load_class_names(path: Path) -> list[str]` — txt/names: 공백 제거한 비어있지 않은 줄; yaml: `names`가 list면 그대로, dict면 int 키 정렬 순

- [ ] **Step 1: Write the failing tests**

```python
def test_collect_items_frame_stride(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=6)          # img_000..005.jpg + 라벨
    items, skipped = collect_items(imgs, lbls, frame_stride=2)
    assert [i.stem for i in items] == ["img_000", "img_002", "img_004"] and skipped == []

def test_collect_items_skips_missing_label_and_bad_image(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=2)
    (lbls/"img_001.txt").unlink(); (imgs/"broken.jpg").write_bytes(b"x"); (lbls/"broken.txt").write_text("")
    items, skipped = collect_items(imgs, lbls, 1)
    assert [i.stem for i in items] == ["img_000"]
    assert sorted(skipped) == [("broken", "unreadable_image"), ("img_001", "missing_label")]

def test_collect_items_duplicate_stem(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=1)
    cv2.imwrite(str(imgs/"img_000.png"), np.zeros((10, 10, 3), np.uint8))
    items, skipped = collect_items(imgs, lbls, 1)
    assert len(items) == 1 and skipped == [("img_000", "duplicate_stem")]

def test_collect_items_excludes_bgswap(tmp_path):
    imgs, lbls = _dataset(tmp_path, n=1)
    cv2.imwrite(str(imgs/"img_000_bgswap1.jpg"), np.zeros((10, 10, 3), np.uint8))
    assert [i.stem for i in collect_items(imgs, lbls, 1)[0]] == ["img_000"]

def test_load_image_bgr_converts_rgba_and_gray(tmp_path):
    cv2.imwrite(str(tmp_path/"a.png"), np.zeros((8, 8, 4), np.uint8))
    cv2.imwrite(str(tmp_path/"g.png"), np.zeros((8, 8), np.uint8))
    assert load_image_bgr(tmp_path/"a.png").shape == (8, 8, 3)
    assert load_image_bgr(tmp_path/"g.png").shape == (8, 8, 3)

@pytest.mark.parametrize("p,expected", [("data/val/images", True), ("data/test", True),
    ("data/train/images", False), ("data/valley/images", False), ("Validation/x", True)])
def test_looks_non_train(p, expected):
    assert looks_non_train(Path(p)) is expected

def test_find_class_names_prefers_classes_txt_in_labels(tmp_path):
    (tmp_path/"lbl").mkdir(); (tmp_path/"img").mkdir()
    (tmp_path/"data.yaml").write_text("names: [a, b]")
    (tmp_path/"lbl"/"classes.txt").write_text("x\ny\n")
    assert find_class_names_file(tmp_path/"img", tmp_path/"lbl") == tmp_path/"lbl"/"classes.txt"

def test_load_class_names_yaml_dict(tmp_path):
    (tmp_path/"data.yaml").write_text("names:\n  1: ball\n  0: robot\n")
    assert load_class_names(tmp_path/"data.yaml") == ["robot", "ball"]
```

`test_pipeline.py`에 추가:
```python
def test_collect_originals_excludes_bgswap(tmp_path):
    (tmp_path/"a.jpg").touch(); (tmp_path/"a_bgswap1.jpg").touch()
    assert [p.name for p in _collect_originals(tmp_path)] == ["a.jpg"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_inputs.py tests/test_pipeline.py -v`
Expected: FAIL (ImportError, `a_bgswap1.jpg` 포함)

- [ ] **Step 3: Implement `inputs.py`, `classes.py`, `_collect_originals` 수정**

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest -v`
Expected: PASS (기존 테스트 포함)

- [ ] **Step 5: Commit**

```bash
git add background_swap/inputs.py background_swap/classes.py core/pipeline.py tests/test_bg_inputs.py tests/test_pipeline.py
git commit -m "feat(background_swap): add input selection, class names, bgswap exclusion"
```

---

### Task 4: 색상 영역 (HSV)

**Files:**
- Create: `augmentor/background_swap/color_region.py`
- Test: `augmentor/tests/test_bg_color_region.py`

**Interfaces:**
- Produces:
  - `hsv_range_from_click(img_bgr: np.ndarray, x: int, y: int, h_tol: int, s_tol: int, v_tol: int) -> tuple[list[int], list[int]]` — 클릭 주변 5×5(경계 클립) HSV 채널별 중앙값 ± tol. S/V는 0–255로 클립, **H는 클립하지 않음**(음수 또는 179 초과로 wrap 표현).
  - `hsv_in_range(hsv: np.ndarray, lower: list[int], upper: list[int]) -> np.ndarray` — bool; H 하한 < 0이면 `[0, upper_h] ∪ [180+lower_h, 179]`, 상한 > 179이면 `[lower_h, 179] ∪ [0, upper_h-180]`
  - `color_mask(img_bgr, lower, upper, largest_only: bool, convex_hull: bool) -> np.ndarray` — `hsv_in_range` → `MORPH_OPEN` 후 `MORPH_CLOSE`(5×5 타원 커널) → largest_only면 최대 연결 성분 → convex_hull이면 각 성분 외곽선의 볼록 껍질을 채움

- [ ] **Step 1: Write the failing tests**

```python
GREEN = (40, 160, 40)   # BGR

def _field_with_line():
    img = np.full((100, 100, 3), (200, 200, 200), np.uint8)
    img[20:80, 10:90] = GREEN
    img[48:52, 10:90] = (255, 255, 255)            # 필드 가로지르는 흰 라인
    return img

def test_hsv_range_from_click_centered_on_green():
    lo, hi = hsv_range_from_click(_field_with_line(), 30, 30, 10, 60, 60)
    h = int(cv2.cvtColor(np.uint8([[GREEN]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    assert lo[0] == h - 10 and hi[0] == h + 10 and 0 <= lo[1] <= hi[1] <= 255

def test_convex_hull_fills_line_inside_field():
    img = _field_with_line(); lo, hi = hsv_range_from_click(img, 30, 30, 10, 60, 60)
    m = color_mask(img, lo, hi, largest_only=True, convex_hull=True)
    assert m[50, 50] and not m[5, 5]

def test_without_hull_line_is_excluded():
    img = _field_with_line(); lo, hi = hsv_range_from_click(img, 30, 30, 10, 60, 60)
    assert not color_mask(img, lo, hi, largest_only=True, convex_hull=False)[50, 50]

def test_largest_only_drops_small_blob():
    img = _field_with_line(); img[86:98, 2:14] = GREEN       # 모폴로지로 지워지지 않는 12px 블롭
    lo, hi = hsv_range_from_click(img, 30, 30, 10, 60, 60)
    assert not color_mask(img, lo, hi, largest_only=True, convex_hull=False)[92, 8]
    assert color_mask(img, lo, hi, largest_only=False, convex_hull=False)[92, 8]

def test_red_hue_wraparound():
    hsv = np.array([[[2, 200, 200], [178, 200, 200], [90, 200, 200]]], np.uint8)
    assert hsv_in_range(hsv, [-5, 100, 100], [5, 255, 255]).tolist() == [[True, True, False]]
    assert hsv_in_range(hsv, [175, 100, 100], [185, 255, 255]).tolist() == [[True, True, False]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_color_region.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `color_region.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_color_region.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/color_region.py tests/test_bg_color_region.py
git commit -m "feat(background_swap): add HSV color region keep masks"
```

---

### Task 5: 마스크 생성 (객체 core/protect, keep, inpaint)

**Files:**
- Create: `augmentor/background_swap/masks.py`
- Create: `augmentor/tests/bg_fakes.py` (가짜 모델 — Task 9에서 확장, 이후 테스트가 공유)
- Test: `augmentor/tests/test_bg_masks.py`

**Interfaces:**
- Consumes: `core.label.BBox`, `domain.MaskSet/Detection`, `settings.BackgroundSwapSettings/KeepItem`, `color_region.color_mask`.
- Produces:
  - `class Segmenter(Protocol)`: `segment_boxes(self, image_rgb: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]` (박스당 HxW bool 1개)
  - `class TextDetector(Protocol)`: `detect(self, image_rgb: np.ndarray, phrases: list[str], box_threshold: float, text_threshold: float) -> list[Detection]`
  - `bbox_to_xyxy(b: BBox, w: int, h: int, margin: float = 0.0) -> tuple[int, int, int, int] | None` — 정규화 → 픽셀, 각 변 `margin` 비율 확장, 이미지로 클립, 폭/높이 0이면 `None`
  - `needs_sam(b: BBox, w: int, h: int, small_box_px: int) -> bool` — 짧은 변(px) ≥ small_box_px
  - `object_core_mask(h: int, w: int, boxes: list[BBox], sam_masks: list[np.ndarray | None], small_box_px: int) -> np.ndarray` — `sam_masks[i]`는 `needs_sam`인 박스만 값, 나머지 `None`. SAM 마스크는 `bbox_to_xyxy(margin=0.1)` 영역으로 클립 → 최대 연결 성분 → 면적 < bbox 면적×0.2면 폴백. 폴백·작은 박스 = `bbox_to_xyxy(margin=0.1)` 사각형.
  - `object_protect_mask(core: np.ndarray, boxes: list[BBox], dilate_px: int, shadow_extend_ratio: float) -> np.ndarray` — `core`를 `(2·dilate_px+1)` 타원 커널로 팽창 ∪ 박스마다 그림자 사각형: x 중심 = bbox 중심, 폭 = bbox 폭×1.2, y는 bbox 하단 ~ 하단+bbox_h×ratio, 이미지로 클립. ratio 0이면 생략.
  - `keep_item_mask(item: KeepItem, img_bgr: np.ndarray, segmenter: Segmenter | None, detector: TextDetector | None) -> tuple[np.ndarray, bool]` — (마스크, 성공). color: `color_mask` 후 면적 비율 ≥ `min_area_ratio`; text: `detector.detect(rgb, [item.prompt], 0.35, 0.25)` 박스들 → `segmenter.segment_boxes` 합집합, 검출 ≥ 1
  - `build_mask_set(img_bgr: np.ndarray, boxes: list[BBox], settings: BackgroundSwapSettings, segmenter: Segmenter | None, detector: TextDetector | None) -> MaskSet` — 필수 항목 실패 → `skip_reason = f"required_keep_missing:{name}"`; 선택 항목 실패 → `warnings += [f"keep_not_found:{name}"]`; `inpaint = ~(object_protect | keep)`; inpaint 비율 < `min_inpaint_ratio` → `skip_reason = "small_inpaint_area"`. SAM은 `needs_sam` 박스가 있을 때만 한 번에 호출.

- [ ] **Step 1: Write the failing tests**

`tests/bg_fakes.py`:
- `FakeSegmenter()` — `segment_boxes`가 박스마다 박스 사각형을 채운 bool 마스크를 반환하고 `calls: list[list[box]]`에 기록
- `FakeDetector(dets_by_phrase: dict[str, list[Detection]])` — `detect`는 phrases 각각의 목록을 이어 붙여 반환
- `GREEN_BGR = (40, 160, 40)`, `GREEN_HSV_RANGE = ([35, 40, 40], [85, 255, 255])`
- `field_image(h=100, w=100) -> np.ndarray` — 회색(128) 배경, 위쪽 30% 아래는 GREEN_BGR

테스트 파일의 `_img()` = `np.full((100, 100, 3), 128, np.uint8)`, `_field_img()` = `field_image()`.

```python
def test_small_box_uses_rect_without_sam():
    b = BBox(0, 0.5, 0.5, 0.1, 0.1)                 # 100x100 이미지에서 45~55px
    assert not needs_sam(b, 100, 100, 32)
    core = object_core_mask(100, 100, [b], [None], 32)
    assert core[50, 50] and core[44, 44] and not core[40, 40]   # 10% 여유: 44~56

def test_sam_mask_too_small_falls_back_to_rect():
    b = BBox(0, 0.5, 0.5, 0.5, 0.5)
    tiny = np.zeros((100, 100), bool); tiny[50, 50] = True
    core = object_core_mask(100, 100, [b], [tiny], 32)
    assert core[30, 30]                               # 사각형 폴백

def test_sam_mask_clipped_to_expanded_box():
    b = BBox(0, 0.5, 0.5, 0.4, 0.4)
    big = np.ones((100, 100), bool)
    core = object_core_mask(100, 100, [b], [big], 32)
    assert core[50, 50] and not core[5, 5]

def test_object_core_clips_and_ignores_degenerate_boxes():
    boxes = [BBox(0, 0.95, 0.5, 0.3, 0.2), BBox(0, 0.5, 0.5, 0.0, 0.2)]
    core = object_core_mask(100, 100, boxes, [None, None], 32)
    assert core.shape == (100, 100) and core[50, 99] and not core[50, 50]

def test_protect_dilation():
    b = BBox(0, 0.5, 0.4, 0.2, 0.2)                  # x 40~60, y 30~50
    core = object_core_mask(100, 100, [b], [None], 32)
    assert object_protect_mask(core, [b], dilate_px=6, shadow_extend_ratio=0.0)[40, 64]
    assert not object_protect_mask(core, [b], dilate_px=0, shadow_extend_ratio=0.0)[40, 64]

def test_protect_shadow_extension():
    b = BBox(0, 0.5, 0.4, 0.2, 0.2)
    core = object_core_mask(100, 100, [b], [None], 32)
    p = object_protect_mask(core, [b], dilate_px=0, shadow_extend_ratio=0.15)   # 폭 1.2배 → x 38~62, y 50~53
    assert p[52, 39] and not p[52, 30] and not p[58, 50]

def test_build_mask_set_required_keep_missing():
    s = BackgroundSwapSettings(keep_items=[KeepItem("fence", "text", required=True, prompt="fence")])
    ms = build_mask_set(_img(), [], s, FakeSegmenter(), FakeDetector({}))
    assert ms.skip_reason == "required_keep_missing:fence"

def test_build_mask_set_optional_keep_missing_warns():
    s = BackgroundSwapSettings(keep_items=[KeepItem("fence", "text", prompt="fence")])
    ms = build_mask_set(_img(), [], s, FakeSegmenter(), FakeDetector({}))
    assert ms.skip_reason is None and ms.warnings == ["keep_not_found:fence"]

def test_build_mask_set_inpaint_is_complement_of_protect():
    lo, hi = GREEN_HSV_RANGE
    s = BackgroundSwapSettings(keep_items=[KeepItem("field", "color", hsv_lower=lo, hsv_upper=hi)])
    ms = build_mask_set(_field_img(), [BBox(0, 0.5, 0.5, 0.2, 0.2)], s, FakeSegmenter(), None)
    assert ms.keep[90, 50] and not ms.keep[5, 50]
    assert (ms.inpaint == ~(ms.object_protect | ms.keep)).all()

def test_build_mask_set_small_inpaint_area_skips():
    ms = build_mask_set(_img(), [BBox(0, 0.5, 0.5, 0.98, 0.98)], BackgroundSwapSettings(), FakeSegmenter(), None)
    assert ms.skip_reason == "small_inpaint_area"

def test_build_mask_set_no_boxes():
    ms = build_mask_set(_img(), [], BackgroundSwapSettings(), None, None)
    assert not ms.object_core.any() and ms.inpaint.all() and ms.skip_reason is None

def test_segmenter_called_once_for_large_boxes_only():
    seg = FakeSegmenter()
    build_mask_set(_img(), [BBox(0, .3, .3, .4, .4), BBox(0, .8, .8, .05, .05)], BackgroundSwapSettings(), seg, None)
    assert len(seg.calls) == 1 and len(seg.calls[0]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_masks.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `masks.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_masks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/masks.py tests/bg_fakes.py tests/test_bg_masks.py
git commit -m "feat(background_swap): add object/keep/inpaint mask generation"
```

---

### Task 6: 합성, 차이 이미지, 오버레이, 작업 해상도

**Files:**
- Create: `augmentor/background_swap/composite.py`
- Create: `augmentor/background_swap/inpaint.py`
- Test: `augmentor/tests/test_bg_composite.py`

**Interfaces:**
- Consumes: `domain.Job/MaskSet`, `settings.BackgroundSwapSettings`.
- Produces (`composite.py`, torch 없음 — GUI도 사용):
  - `alpha_map(protect: np.ndarray, feather_px: int) -> np.ndarray` — float32; protect 내부 1.0, 바깥은 `cv2.distanceTransform(~protect)` 거리 d에 대해 `max(0, 1 - d / (feather_px + 1))`, feather_px 0이면 protect 그대로
  - `composite(orig: np.ndarray, gen: np.ndarray, protect: np.ndarray, feather_px: int) -> np.ndarray` — uint8, protect 내부는 `orig`를 그대로 복사(부동소수 경유 금지)
  - `diff_image(orig, out, protect) -> np.ndarray` — BGR; `|out-orig|` 채널 최대값 그레이, protect 내부에서 차이 > 0인 픽셀은 `(0, 0, 255)`
  - `mask_overlay(img_bgr, object_core, object_protect, keep, alpha: float = 0.45) -> np.ndarray` — 빨강 `(0,0,255)`=core, 주황 `(0,140,255)`=protect−core, 초록 `(0,200,0)`=keep, 나머지 원본
- Produces (`inpaint.py`, torch 없음):
  - `class Inpainter(Protocol)`: `native_px: int`; `inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str, negative_prompt: str, seed: int) -> np.ndarray` — 입력 크기 그대로 RGB 반환
  - `working_size(w: int, h: int, native_px: int, max_px: int, mode: str, manual_px: int) -> tuple[int, int]` — 긴 변 목표 = manual이면 `manual_px`, auto면 `min(max(long, native_px), max_px)`; 비율 유지, 각 변 `max(8, round(x / 8) * 8)`
  - `run_inpaint(inpainter: Inpainter, orig_bgr: np.ndarray, masks: MaskSet, job: Job, settings: BackgroundSwapSettings) -> np.ndarray` — 작업 크기로 리사이즈(이미지 `INTER_AREA`/`INTER_CUBIC`, 마스크 `INTER_NEAREST`) → `inpaint` → 원본 크기로 리사이즈 → `composite(orig, gen, masks.protect, settings.feather_px)` 반환 (BGR)

- [ ] **Step 1: Write the failing tests**

```python
def test_composite_protect_is_bytewise_original():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8); gen = 255 - orig
    protect = np.zeros((60, 80), bool); protect[10:30, 20:50] = True
    out = composite(orig, gen, protect, feather_px=4)
    assert (out[protect] == orig[protect]).all()

def test_feather_only_on_inpaint_side():
    orig = np.zeros((40, 40, 3), np.uint8); gen = np.full_like(orig, 200)
    protect = np.zeros((40, 40), bool); protect[:, :20] = True
    out = composite(orig, gen, protect, feather_px=4)
    assert 0 < out[20, 21, 0] < 200          # 경계 바로 바깥은 섞임
    assert out[20, 30, 0] == 200             # feather 밖은 생성 그대로

def test_diff_image_marks_protected_changes_red():
    orig = np.zeros((10, 10, 3), np.uint8); out = orig.copy(); out[2, 2] = 50
    protect = np.zeros((10, 10), bool); protect[2, 2] = True
    assert tuple(diff_image(orig, out, protect)[2, 2]) == (0, 0, 255)

@pytest.mark.parametrize("w,h,native,mx,mode,manual,expected", [
    (640, 480, 1024, 1280, "auto", 1024, (1024, 768)),    # 작은 원본 → 네이티브로 업스케일
    (1920, 1080, 1024, 1280, "auto", 1024, (1280, 720)),  # 큰 원본 → 상한으로 다운스케일
    (1100, 700, 1024, 1280, "auto", 1024, (1104, 704)),   # 범위 안 → 원본 (8 배수)
    (640, 480, 512, 1280, "auto", 1024, (640, 480)),      # SD1.5
    (640, 480, 1024, 1280, "manual", 768, (768, 576)),
    (1024, 768, 1024, 768, "auto", 1024, (768, 576)),     # native > max → max 우선
])
def test_working_size(w, h, native, mx, mode, manual, expected):
    assert working_size(w, h, native, mx, mode, manual) == expected

def test_run_inpaint_returns_original_size_and_preserves_protect():
    class FakeInpainter:
        native_px = 64
        def __init__(self): self.sizes = []
        def inpaint(self, image_rgb, mask, prompt, negative_prompt, seed):
            self.sizes.append(image_rgb.shape[:2]); return np.full_like(image_rgb, 123)
    orig = np.random.default_rng(0).integers(0, 256, (30, 50, 3), dtype=np.uint8)
    protect = np.zeros((30, 50), bool); protect[5:15, 5:20] = True
    masks = MaskSet(protect, protect, np.zeros_like(protect), ~protect, [], None)
    job = Job("a", 1, 1, 0, {}, "p", "n"); settings = BackgroundSwapSettings(max_resolution_px=128)
    fake = FakeInpainter()
    out = run_inpaint(fake, orig, masks, job, settings)
    assert out.shape == orig.shape and fake.sizes == [(40, 64)]
    assert (out[masks.protect] == orig[masks.protect]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_composite.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `composite.py`, `inpaint.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_composite.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/composite.py background_swap/inpaint.py tests/test_bg_composite.py
git commit -m "feat(background_swap): add compositing, diff view, working resolution"
```

---

### Task 7: 장면 프롬프트

**Files:**
- Create: `augmentor/background_swap/prompts.py`
- Test: `augmentor/tests/test_bg_prompts.py`

**Interfaces:**
- Produces:
  - `@dataclass PromptRequest(scene: dict[str, str], keep_names: list[str], seed: int)`
  - `class PromptGenerator(Protocol)`: `generate_batch(self, requests: list[PromptRequest]) -> list[str]`
  - `derive_seed(base_seed: int, stem: str, variant: int, attempt: int) -> int` — `sha256(f"{base_seed}:{stem}:{variant}:{attempt}")` 앞 8바이트 big-endian int `% 2**31`
  - `sample_scene(axes: dict[str, list[str]], seed: int) -> dict[str, str]` — `random.Random(seed)`, 축 삽입 순서대로 `choice`
  - `TemplatePromptGenerator()` — `", ".join(scene 값들)` + (keep_names 있으면 `", surrounding " + " and ".join(keep_names)`) + `", photorealistic"`
  - `QUALITY_NEGATIVES = ["blurry", "distorted", "text", "watermark", "extra objects"]`
  - `negative_prompt(class_names: list[str]) -> str` — `", ".join(class_names + QUALITY_NEGATIVES)`
  - `sanitize_prompt(text: str, class_names: list[str]) -> str` — 클래스 이름을 대소문자 무시 단어 경계(`\b`)로 제거, 연속 공백/쉼표 정리, 첫 줄만, 앞뒤 따옴표 제거, 60단어 초과 시 자름
  - `LLM_SYSTEM_PROMPT: str` — 영문: 한 문장 ≤60 단어의 SDXL 인페인팅용 실내 대회장 배경 묘사만 출력, 사람·로봇·공·동물·차량 등 셀 수 있는 물체 및 사용자 제공 금지어 언급 금지, 유지 대상은 "surrounding ..." 맥락으로만
  - `build_llm_messages(req: PromptRequest, class_names: list[str]) -> list[dict[str, str]]` — `[{"role": "system", ...}, {"role": "user", ...}]`, user에 축 값·keep 이름·금지어(class_names) 포함

- [ ] **Step 1: Write the failing tests**

```python
def test_derive_seed_stable_and_distinct():
    assert derive_seed(0, "a", 1, 1) == derive_seed(0, "a", 1, 1)
    assert len({derive_seed(0, "a", v, k) for v in (1, 2) for k in (1, 2)}) == 4
    assert 0 <= derive_seed(0, "a", 1, 1) < 2**31

def test_sample_scene_deterministic_one_value_per_axis():
    axes = {"venue": ["a", "b", "c"], "lighting": ["x", "y"]}
    s = sample_scene(axes, 42)
    assert s == sample_scene(axes, 42) and list(s) == ["venue", "lighting"]
    assert s["venue"] in axes["venue"]

def test_template_prompt():
    req = PromptRequest({"venue": "sports hall", "lighting": "warm LED"}, ["green field mat", "fence"], 0)
    assert TemplatePromptGenerator().generate_batch([req]) == [
        "sports hall, warm LED, surrounding green field mat and fence, photorealistic"]

def test_sanitize_removes_class_names_whole_word():
    out = sanitize_prompt('"A hall with a Robot and robotics banners, ball pit"', ["robot", "ball"])
    assert "Robot" not in out and "robotics" in out and " ball " not in f" {out} "
    assert not out.startswith('"')

def test_sanitize_truncates_to_60_words():
    assert len(sanitize_prompt(" ".join(["w"] * 80), []).split()) == 60

def test_negative_prompt():
    assert negative_prompt(["robot"]) == "robot, blurry, distorted, text, watermark, extra objects"

def test_llm_messages_include_forbidden_words():
    msgs = build_llm_messages(PromptRequest({"venue": "gym"}, ["mat"], 0), ["robot"])
    assert msgs[0]["role"] == "system" and "robot" in msgs[1]["content"] and "gym" in msgs[1]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_prompts.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `prompts.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/prompts.py tests/test_bg_prompts.py
git commit -m "feat(background_swap): add scene sampling and prompt generation"
```

---

### Task 8: 검증 로직

**Files:**
- Create: `augmentor/background_swap/validate.py`
- Test: `augmentor/tests/test_bg_validate.py`

**Interfaces:**
- Consumes: `domain.Detection/ValidationResult`.
- Produces:
  - `iou(a: tuple[float, ...], b: tuple[float, ...]) -> float` (xyxy)
  - `rule_checks(orig: np.ndarray, out: np.ndarray, protect: np.ndarray, inpaint: np.ndarray, min_change: float) -> tuple[list[str], dict[str, object]]` — reasons `"size_mismatch"`(이후 검사 생략), `"protected_changed"`, `"too_little_change"`; checks `{"protected_identical": bool, "mean_change": float}`. mean_change = inpaint 픽셀의 채널 평균 절대차 평균(inpaint 비어있으면 0.0).
  - `find_new_objects(dets: list[Detection], label_boxes: list[tuple[int, int, int, int]], inpaint: np.ndarray, score_thr: float = 0.35, iou_thr: float = 0.3, inpaint_frac: float = 0.5) -> list[Detection]`
  - `VLM_PROMPT: str` — 영문: bbox가 표시된 이미지에 대해 JSON 한 객체만 출력 `{"natural_indoor_venue": bool, "boundary_artifacts": bool, "new_objects": bool, "reason": str}`; new_objects = 표시된 박스 밖에 라벨 대상과 같은 종류의 물체가 보이는지(클래스 이름 목록 삽입 자리 `{class_names}`)
  - `parse_vlm_json(text: str) -> dict | None` — 첫 `{`~대응 `}` 추출 → `json.loads`, 세 bool 키가 모두 bool이 아니면 `None`
  - `class ImageValidator(Protocol)`: `judge_batch(self, images_rgb: list[np.ndarray], prompt: str) -> list[str]`
  - `combine(rule_reasons: list[str], checks: dict, gdino_new: list[Detection] | None, vlm_text: str | None) -> ValidationResult` — `gdino_new` None = 검사 안 함, `vlm_text` None = VLM 안 씀. 사유 추가 순서: rule → `"new_object:gdino"` → VLM(`"vlm_unparsable"` 또는 `"vlm_unnatural"`, `"vlm_boundary_artifacts"`, `"new_object:vlm"`). checks에 `"gdino_new_count"`, `"vlm"`(파싱된 dict 또는 원문) 기록.

- [ ] **Step 1: Write the failing tests**

```python
def test_rule_checks_ok():
    orig = np.zeros((10, 10, 3), np.uint8); out = orig.copy(); out[:, 5:] = 100
    protect = np.zeros((10, 10), bool); protect[:, :5] = True
    reasons, checks = rule_checks(orig, out, protect, ~protect, 20.0)
    assert reasons == [] and checks["protected_identical"] and checks["mean_change"] == 100.0

def test_rule_checks_detects_protected_change_and_little_change():
    orig = np.zeros((10, 10, 3), np.uint8); out = orig.copy(); out[0, 0] = 1; out[:, 5:] = 5
    protect = np.zeros((10, 10), bool); protect[:, :5] = True
    assert rule_checks(orig, out, protect, ~protect, 20.0)[0] == ["protected_changed", "too_little_change"]

def test_rule_checks_size_mismatch_short_circuits():
    assert rule_checks(np.zeros((10, 10, 3), np.uint8), np.zeros((9, 10, 3), np.uint8),
                       np.zeros((10, 10), bool), np.ones((10, 10), bool), 20.0)[0] == ["size_mismatch"]

def test_find_new_objects_filters_by_iou_score_and_region():
    inpaint = np.zeros((100, 100), bool); inpaint[:, 50:] = True
    labels = [(0, 0, 20, 20)]
    dets = [Detection((0, 0, 20, 20), 0.9, "robot"),     # 기존 라벨과 겹침
            Detection((60, 60, 80, 80), 0.3, "robot"),   # 점수 낮음
            Detection((10, 60, 30, 80), 0.9, "robot"),   # 보호 영역
            Detection((60, 10, 80, 30), 0.9, "robot")]   # 새 객체
    assert find_new_objects(dets, labels, inpaint) == [dets[3]]

@pytest.mark.parametrize("text,ok", [
    ('Sure: {"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": false, "reason": "ok"}', True),
    ('{"natural_indoor_venue": "yes", "boundary_artifacts": false, "new_objects": false}', False),
    ('no json here', False)])
def test_parse_vlm_json(text, ok):
    assert (parse_vlm_json(text) is not None) is ok

def test_combine_orders_reasons_and_ors_new_objects():
    vlm = '{"natural_indoor_venue": false, "boundary_artifacts": true, "new_objects": true, "reason": "x"}'
    r = combine([], {}, [Detection((0, 0, 1, 1), 0.9, "robot")], vlm)
    assert not r.accepted
    assert r.reasons == ["new_object:gdino", "vlm_unnatural", "vlm_boundary_artifacts", "new_object:vlm"]

def test_combine_accepts_when_all_skipped_or_clean():
    assert combine([], {}, None, None).accepted
    assert combine([], {}, [], '{"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": false}').accepted

def test_combine_unparsable_vlm_rejects():
    assert combine([], {}, None, "garbage").reasons == ["vlm_unparsable"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_validate.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `validate.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bg_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/validate.py tests/test_bg_validate.py
git commit -m "feat(background_swap): add rule, new-object, and VLM validation"
```

---

### Task 9: 워크플로 (단계, 재시도, 재개, Manifest, 미리보기)

**Files:**
- Create: `augmentor/background_swap/workflow.py`
- Create: `augmentor/tests/bg_fakes.py` (가짜 모델 팩토리 — Task 10, 11 테스트도 사용)
- Test: `augmentor/tests/test_bg_workflow.py`

**Interfaces:**
- Consumes: Task 1–8 전부. `core.label.read_labels`, `AugmentTab._draw_bboxes`와 같은 방식의 bbox 그리기(여기서는 `composite.py`에 `draw_boxes(img_bgr, boxes: list[BBox]) -> np.ndarray` 추가 — 클래스별 색 `_BBOX_COLORS` 동일 값 복사).
- Produces:
  - `class ModelFactory(Protocol)` — 각 메서드는 컨텍스트 매니저(종료 시 GPU 해제):
    `mask_models() -> ContextManager[tuple[Segmenter, TextDetector]]`, `prompt_model() -> ContextManager[PromptGenerator]`, `inpainter() -> ContextManager[Inpainter]`, `detector() -> ContextManager[TextDetector]`, `validator() -> ContextManager[ImageValidator]`
  - `class BackgroundSwapWorkflow(settings: BackgroundSwapSettings, factory: ModelFactory, progress: Progress, class_names: list[str], only_stems: list[str] | None = None)`
    - `run() -> None` — 아래 순서. 단계 이름(`progress.stage`): `"inputs"`, `"masks"`, `"prompts"`, `"inpaint"`, `"validate"`, `"report"`
    - 결과 경로 헬퍼: `generated_path(stem, n, k)`, `accepted_image_path(stem, n)`, `rejected_path(stem, n, k)`
  - `run_preview(settings, factory, progress, class_names, n: int, rng_seed: int) -> Path` — `output_dir/_preview`를 지우고, `num_variants=1`, `max_attempts=1`, `output_dir=_preview`로 복사한 설정과 무작위 `n`개 stem(`random.Random(rng_seed)`)으로 `BackgroundSwapWorkflow(...).run()` 후 `_preview` 경로 반환
  - Manifest 레코드 스키마: `{"stem", "variant", "attempt", "seed", "scene", "prompt", "status", "reasons", "checks", "warnings"}` (skipped는 `variant/attempt = null`)

**run() 흐름 (구현자가 정할 수 없는 결정):**
1. inputs: `collect_items` → `items.jsonl` 재작성. skipped는 Manifest에 아직 같은 stem의 skipped 레코드가 없을 때만 추가.
2. masks: `masks/{stem}.json`(`{"warnings", "skip_reason"}`)이 없는 항목만 처리. 처리할 항목이 있을 때만 `factory.mask_models()`를 연다. `color`만 있고 SAM 필요 박스가 없어도 동일하게 연다(단순화). skip이면 Manifest에 skipped 기록, 마스크 PNG는 skip이 아닐 때만 `object_core/object_protect/keep/inpaint` 4개 저장. json은 PNG 저장 후 마지막에 써서 재개 기준으로 삼는다.
3. GDINO recall: `class_names`가 있으면 mask 단계 안에서 같은 detector로 원본마다 검출해 `report.class_recall` 입력 누적(Task 10) → `recall.json` 저장. 이미 있으면 생략.
4. 라운드 `k = 1..max_attempts`: 대상 = 비-skip 항목 × variant `1..num_variants` 중 Manifest에 accepted가 없고, 이번 attempt 레코드도 없는 것. 대상이 없으면 종료.
   - prompts: `jobs.jsonl`에 (stem, n, k)가 없는 대상만. `seed = derive_seed(settings.seed, stem, n, k)`, `scene = sample_scene(...)`, `use_llm`이면 `factory.prompt_model()`, 아니면 `TemplatePromptGenerator`. `prompt = sanitize_prompt(...)`, `negative_prompt(class_names)`. 배치 크기 8.
   - inpaint: `generated_path`가 없는 job만, `factory.inpainter()` 안에서 `run_inpaint` → `imwrite_atomic`.
   - validate: Manifest 레코드가 없는 job만. rule_checks 전부 → `class_names` 있으면 `factory.detector()`로 `find_new_objects` (phrases = class_names, box 0.35 / text 0.25) → `use_vlm`이면 `factory.validator()`로 bbox 표시본에 `judge_batch` (배치 4) → `combine`. accepted: `Accepted/images/{stem}_bgswap{n}.jpg`, `Accepted/labels/{stem}_bgswap{n}.txt`(원본 라벨 복사), `Accepted/previews/…jpg`(bbox 표시본). rejected: `Rejected/{stem}_bgswap{n}_a{k}.jpg`. 그다음 Manifest 추가(파일 먼저, 레코드 나중).
5. 라운드 후 accepted가 없는 (stem, n) 중 exhausted 레코드가 없는 것에 `status="exhausted"` 기록.
6. `report.write_report(output_dir)` (Task 10). 테스트에서는 Task 10 전까지 이 호출을 넣지 않고, Task 10에서 추가한다.

`tests/bg_fakes.py`에 추가 (테스트에서는 `from tests.bg_fakes import *`, `_settings = make_settings`, `_vlm = vlm_json`):
- `FakeInpainter(native_px=64, fail_after: int | None = None, raise_exc: Exception | None = None)` — 입력을 반전(`255 - img`)해 반환, `calls` 카운트; `fail_after`번 성공 후 `RuntimeError`, `raise_exc`면 첫 호출에 그 예외
- `FakeValidator(responses: list[str])` — `judge_batch`가 이미지마다 다음 응답 반환(소진 시 마지막 반복)
- `FakeFactory(segmenter, detector, prompt_gen, inpainter, validator)` — 각 컨텍스트 진입 시 메서드 이름을 `opened: list[str]`에 기록, `inpainter_obj` 속성 노출. `FakeFactory.default(**overrides)`: `FakeSegmenter()`, `FakeDetector({})`, `TemplatePromptGenerator()`, `FakeInpainter()`, `FakeValidator([vlm_json(True)])`
- `vlm_json(ok: bool) -> str` — 세 키 모두 정상(ok) 또는 `natural_indoor_venue=false`
- `make_settings(tmp_path, n_images=1, size=(64, 48), rgba=False, **overrides) -> BackgroundSwapSettings` — `tmp_path/"train"/"images"`, `…/"labels"`에 `img_000…` 이미지(무작위 노이즈, seed 고정)와 라벨 `"0 0.5 0.5 0.3 0.3\n"`을 만들고 `output_dir=tmp_path/"out"`; overrides는 settings 필드

- [ ] **Step 1: Write the failing tests**

```python
def test_run_produces_accepted_with_original_labels(tmp_path):
    s = _settings(tmp_path, n_images=2, num_variants=2, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    acc = sorted(p.name for p in (s.output_dir/"Accepted"/"images").iterdir())
    assert acc == ["img_000_bgswap1.jpg", "img_000_bgswap2.jpg", "img_001_bgswap1.jpg", "img_001_bgswap2.jpg"]
    assert (s.output_dir/"Accepted"/"labels"/"img_000_bgswap1.txt").read_text() == (s.labels_dir/"img_000.txt").read_text()

def test_rejected_then_retried_until_exhausted(tmp_path):
    bad = '{"natural_indoor_venue": false, "boundary_artifacts": false, "new_objects": false}'
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=3)
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([bad])), NullProgress(), []).run()
    m = read_jsonl(s.output_dir/"Manifest.jsonl")
    assert [r["status"] for r in m] == ["rejected", "rejected", "rejected", "exhausted"]
    assert len({r["seed"] for r in m if r["status"] == "rejected"}) == 3
    assert len(list((s.output_dir/"Rejected").iterdir())) == 3

def test_retry_stops_after_accept(tmp_path):
    bad, good = _vlm(False), _vlm(True)
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=3)
    BackgroundSwapWorkflow(s, FakeFactory.default(validator=FakeValidator([bad, good])), NullProgress(), []).run()
    assert [r["status"] for r in read_jsonl(s.output_dir/"Manifest.jsonl")] == ["rejected", "accepted"]

def test_resume_skips_completed_work(tmp_path):
    s = _settings(tmp_path, n_images=2, num_variants=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    f2 = FakeFactory.default()
    BackgroundSwapWorkflow(s, f2, NullProgress(), []).run()
    assert f2.inpainter_obj.calls == 0 and f2.opened == []
    assert len(read_jsonl(s.output_dir/"Manifest.jsonl")) == 2

def test_resume_after_crash_mid_inpaint(tmp_path):
    s = _settings(tmp_path, n_images=3, num_variants=1, use_vlm=False)
    crashing = FakeFactory.default(inpainter=FakeInpainter(fail_after=1))
    with pytest.raises(RuntimeError):
        BackgroundSwapWorkflow(s, crashing, NullProgress(), []).run()
    f2 = FakeFactory.default()
    BackgroundSwapWorkflow(s, f2, NullProgress(), []).run()
    assert f2.inpainter_obj.calls == 2
    assert len(list((s.output_dir/"Accepted"/"images").iterdir())) == 3

def test_skipped_items_recorded_once(tmp_path):
    s = _settings(tmp_path, n_images=1, keep_items=[KeepItem("fence", "text", required=True, prompt="fence")])
    for _ in range(2):
        BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    m = read_jsonl(s.output_dir/"Manifest.jsonl")
    assert [(r["status"], r["reasons"]) for r in m] == [("skipped", ["required_keep_missing:fence"])]

def test_class_names_enable_gdino_new_object_check(tmp_path):
    det = FakeDetector({"robot": [Detection((1, 1, 5, 5), 0.9, "robot")]})   # 배경 영역 새 객체
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(detector=det), NullProgress(), ["robot"]).run()
    r = read_jsonl(s.output_dir/"Manifest.jsonl")[0]
    assert r["status"] == "rejected" and "new_object:gdino" in r["reasons"]

def test_use_llm_false_does_not_open_prompt_model(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_llm=False, use_vlm=False)
    f = FakeFactory.default(); BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert "prompt_model" not in f.opened and "validator" not in f.opened

def test_each_model_context_opened_once_per_round(tmp_path):
    s = _settings(tmp_path, n_images=3, num_variants=2, max_attempts=1)
    f = FakeFactory.default(); BackgroundSwapWorkflow(s, f, NullProgress(), []).run()
    assert f.opened == ["mask_models", "prompt_model", "inpainter", "validator"]

def test_rgba_input_outputs_three_channel_jpg(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False, rgba=True)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    out = cv2.imread(str(s.output_dir/"Accepted"/"images"/"img_000_bgswap1.jpg"), cv2.IMREAD_UNCHANGED)
    assert out.shape[2] == 3

def test_run_preview_uses_isolated_dir(tmp_path):
    s = _settings(tmp_path, n_images=5, num_variants=3)
    d = run_preview(s, FakeFactory.default(), NullProgress(), [], n=2, rng_seed=0)
    assert d == s.output_dir/"_preview"
    assert len(read_jsonl(d/"Manifest.jsonl")) == 2 and not (s.output_dir/"Manifest.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_workflow.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Extend `tests/bg_fakes.py`, implement `workflow.py` and `composite.draw_boxes`**

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/workflow.py background_swap/composite.py tests/bg_fakes.py tests/test_bg_workflow.py
git commit -m "feat(background_swap): add staged resumable workflow with retries"
```

---

### Task 10: 리포트 + Export

**Files:**
- Create: `augmentor/background_swap/report.py`
- Create: `augmentor/background_swap/export.py`
- Modify: `augmentor/background_swap/workflow.py` (`run()` 끝에 `"report"` 단계, mask 단계의 recall 누적을 `class_recall` 사용으로 연결)
- Test: `augmentor/tests/test_bg_report.py`

**Interfaces:**
- Consumes: `store.read_jsonl`, `inputs.looks_non_train`, `validate.iou`.
- Produces (`report.py`):
  - `class_recall(label_boxes: list[tuple[int, tuple[int, int, int, int]]], dets: list[Detection], class_names: list[str]) -> dict[str, list[int]]` — `{name: [hits, total]}`; 라벨 (cls, xyxy)가 같은 이름(`class_names[cls]`) 검출과 IoU ≥ 0.5면 hit
  - `merge_recall(a, b) -> dict[str, list[int]]`
  - `build_report(output_dir: Path) -> dict` — 키: `"status_counts"`, `"reason_counts"`, `"scene_counts": {"all": {axis: {value: n}}, "accepted": {...}}` (rejected/accepted 레코드 기준), `"class_recall": {name: {"hits", "total", "recall"}}` (`recall.json` 있으면), `"stage_seconds"`(`timings.json` 있으면), `"note": "GDINO 미검출은 새 객체가 없다는 뜻이 아닙니다."`
  - `write_report(output_dir: Path) -> dict` (`report.json` 원자적 쓰기)
- Produces (`export.py`):
  - `class NonTrainTargetError(ValueError)`
  - `@dataclass ExportResult(copied: int, skipped_existing: int)`
  - `export_accepted(output_dir: Path, images_dir: Path, labels_dir: Path, allow_non_train: bool) -> ExportResult` — 대상 경로가 `looks_non_train`이고 허용 안 되면 raise. 이미지/라벨 중 하나라도 대상에 있으면 쌍 전체 건너뜀.
- workflow: 각 단계 소요 시간을 `timings.json`에 누적(`{stage: seconds}` 합산).

- [ ] **Step 1: Write the failing tests**

```python
def test_class_recall():
    labels = [(0, (0, 0, 10, 10)), (0, (50, 50, 60, 60)), (1, (20, 20, 30, 30))]
    dets = [Detection((0, 0, 10, 9), 0.9, "robot"), Detection((20, 20, 30, 30), 0.9, "robot")]
    assert class_recall(labels, dets, ["robot", "ball"]) == {"robot": [1, 2], "ball": [0, 1]}

def test_build_report_counts(tmp_path):
    for r in [_rec("accepted", scene={"venue": "gym"}), _rec("rejected", ["vlm_unnatural"], scene={"venue": "hall"}),
              _rec("skipped", ["small_inpaint_area"])]:
        append_jsonl(tmp_path/"Manifest.jsonl", r)
    rep = build_report(tmp_path)
    assert rep["status_counts"] == {"accepted": 1, "rejected": 1, "skipped": 1}
    assert rep["reason_counts"] == {"vlm_unnatural": 1, "small_inpaint_area": 1}
    assert rep["scene_counts"]["accepted"] == {"venue": {"gym": 1}}
    assert rep["scene_counts"]["all"] == {"venue": {"gym": 1, "hall": 1}}

def test_export_copies_and_skips_existing(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1", "a_bgswap2"])
    imgs, lbls = tmp_path/"train"/"images", tmp_path/"train"/"labels"; imgs.mkdir(parents=True); lbls.mkdir()
    (imgs/"a_bgswap2.jpg").write_bytes(b"old")
    assert export_accepted(out, imgs, lbls, False) == ExportResult(copied=1, skipped_existing=1)
    assert (imgs/"a_bgswap2.jpg").read_bytes() == b"old"

def test_export_refuses_val_target(tmp_path):
    out = _accepted(tmp_path, ["a_bgswap1"])
    with pytest.raises(NonTrainTargetError):
        export_accepted(out, tmp_path/"val"/"images", tmp_path/"val"/"labels", False)

def test_workflow_writes_report(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False)
    BackgroundSwapWorkflow(s, FakeFactory.default(), NullProgress(), []).run()
    assert json.loads((s.output_dir/"report.json").read_text())["status_counts"] == {"accepted": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_report.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `report.py`, `export.py`, workflow 연결**

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add background_swap/report.py background_swap/export.py background_swap/workflow.py tests/test_bg_report.py
git commit -m "feat(background_swap): add run report and accepted export"
```

---

### Task 11: CLI 워커

**Files:**
- Create: `augmentor/background_swap/cli.py`
- Create: `augmentor/background_swap/__main__.py` (`raise SystemExit(main(sys.argv[1:]))`)
- Test: `augmentor/tests/test_bg_cli.py`

**Interfaces:**
- Consumes: `settings.load_settings/validate_settings/generation_diff`, `classes.*`, `inputs.looks_non_train`, `workflow.*`, `export.export_accepted`, `progress.JsonLinesProgress`, `domain.GpuOutOfMemory`.
- Produces:
  - `main(argv: list[str], factory_builder: Callable[[BackgroundSwapSettings], ModelFactory] | None = None, stream=sys.stdout) -> int`
    - 서브커맨드: `run --settings PATH`, `preview --settings PATH [--n 3] [--seed 0]`, `export --settings PATH --images-dir D --labels-dir D`
    - 종료 코드: 0 성공, 2 설정 오류, 3 non-train 거부, 4 기존 출력과 설정 불일치, 5 의존성/모델 사전 점검 실패, 6 GPU OOM, 1 기타 예외
    - 모든 실패는 `{"event": "error", "code": <int>, "message": <한국어>}` 한 줄 출력 후 해당 코드 반환. 성공 시 `{"event": "done"}` (preview는 `{"event": "done", "preview_dir": "..."}`, export는 `{"event": "done", "copied": n, "skipped_existing": m}`)
  - 동작 규칙:
    - `validate_settings` 오류 → 2 (메시지 `"; ".join`)
    - `run`/`preview`: `images_dir`가 `looks_non_train`이고 `allow_non_train` 아님 → 3
    - `run`: `--settings` 경로가 `output_dir/run_settings.json`과 다른 파일이고 그 파일이 존재하며 `generation_diff`가 비어있지 않으면 → 4. 같은 파일이면 통과(GUI가 비교·결정 후 기록함)
    - 클래스 이름: `class_names_file` 또는 `find_class_names_file` → `load_class_names`, 없으면 `[]` + `{"event": "warning", "message": "클래스 이름 파일이 없어 새 객체 검사를 건너뜁니다."}`
    - `factory_builder` None이면 `from background_swap.models import HFModelFactory, preflight` 지연 import — `ImportError` → 5 (`"pip install -r requirements-bg.txt"` 안내 포함), `preflight(settings)` 오류 목록 비어있지 않으면 → 5
    - `GpuOutOfMemory` → 6, 메시지 `"VRAM 부족 — 고급 설정에서 해상도를 낮추거나 더 작은 모델을 사용하세요."`

- [ ] **Step 1: Write the failing tests**

```python
def _events(buf): return [json.loads(l) for l in buf.getvalue().splitlines()]

def test_run_success_emits_progress_and_done(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, use_vlm=False); p = s.output_dir/"run_settings.json"
    save_settings(s, p); buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda _: FakeFactory.default(), buf) == 0
    ev = _events(buf); assert ev[-1] == {"event": "done"} and any(e.get("stage") == "inpaint" for e in ev)

def test_invalid_settings_exit_2(tmp_path):
    p = tmp_path/"s.json"; save_settings(BackgroundSwapSettings(), p); buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda _: FakeFactory.default(), buf) == 2
    assert _events(buf)[-1]["event"] == "error"

def test_non_train_exit_3(tmp_path):
    s = _settings(tmp_path/"val", n_images=1); p = tmp_path/"s.json"; save_settings(s, p)
    assert main(["run", "--settings", str(p)], lambda _: FakeFactory.default(), io.StringIO()) == 3

def test_settings_mismatch_exit_4(tmp_path):
    s = _settings(tmp_path, n_images=1); save_settings(s, s.output_dir/"run_settings.json")
    other = tmp_path/"other.json"; save_settings(dataclasses.replace(s, seed=9), other)
    assert main(["run", "--settings", str(other)], lambda _: FakeFactory.default(), io.StringIO()) == 4

def test_oom_exit_6(tmp_path):
    s = _settings(tmp_path, n_images=1); p = s.output_dir/"run_settings.json"; save_settings(s, p)
    f = FakeFactory.default(inpainter=FakeInpainter(raise_exc=GpuOutOfMemory()))
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], lambda _: f, buf) == 6
    assert "VRAM" in _events(buf)[-1]["message"]

def test_missing_heavy_deps_exit_5(tmp_path, monkeypatch):
    s = _settings(tmp_path, n_images=1); p = s.output_dir/"run_settings.json"; save_settings(s, p)
    monkeypatch.setitem(sys.modules, "background_swap.models", None)    # import 시 ImportError
    buf = io.StringIO()
    assert main(["run", "--settings", str(p)], None, buf) == 5
    assert "requirements-bg.txt" in _events(buf)[-1]["message"]

def test_preview_and_export(tmp_path):
    s = _settings(tmp_path, n_images=2, use_vlm=False); p = s.output_dir/"run_settings.json"; save_settings(s, p)
    buf = io.StringIO()
    assert main(["preview", "--settings", str(p), "--n", "1"], lambda _: FakeFactory.default(), buf) == 0
    assert _events(buf)[-1]["preview_dir"].endswith("_preview")
    main(["run", "--settings", str(p)], lambda _: FakeFactory.default(), io.StringIO())
    dst = tmp_path/"train"; (dst/"images").mkdir(parents=True); (dst/"labels").mkdir()
    buf = io.StringIO()
    assert main(["export", "--settings", str(p), "--images-dir", str(dst/"images"), "--labels-dir", str(dst/"labels")],
                lambda _: FakeFactory.default(), buf) == 0
    assert _events(buf)[-1]["copied"] == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_cli.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `cli.py`, `__main__.py`**

- [ ] **Step 4: Run all tests + module entry check**

Run: `python3 -m pytest -v && python3 -m background_swap --help`
Expected: PASS, 서브커맨드 3개가 보이는 도움말

- [ ] **Step 5: Commit**

```bash
git add background_swap/cli.py background_swap/__main__.py tests/test_bg_cli.py
git commit -m "feat(background_swap): add CLI worker with JSON events and exit codes"
```

---

### Task 12: 실제 HF 모델 래퍼 + GPU 스모크

**Files:**
- Create: `augmentor/background_swap/models.py`
- Create: `augmentor/pytest.ini`
- Test: `augmentor/tests/test_bg_gpu_smoke.py`

**Interfaces:**
- Consumes: Protocol들(`Segmenter`, `TextDetector`, `PromptGenerator`, `Inpainter`, `ImageValidator`, `ModelFactory`), `prompts.build_llm_messages/LLM_SYSTEM_PROMPT`, `validate.VLM_PROMPT`, `domain.Detection/GpuOutOfMemory`, Task 0의 `LLM_LOAD/VLM_LOAD`.
- Produces:
  - `HFModelFactory(settings: BackgroundSwapSettings, class_names: list[str])` — `ModelFactory` 구현. 각 컨텍스트 종료 시 `release(*objs)`: 참조 삭제 → `gc.collect()` → `torch.cuda.empty_cache()`
  - `preflight(settings) -> list[str]` — CUDA 사용 가능 여부, 5개 모델 ID 각각 `huggingface_hub.try_to_load_from_cache(id, "config.json")` 또는 `model_index.json`(inpainter) 캐시 확인, 없으면 `huggingface_hub.model_info(id)`로 접근 확인; 실패 ID를 한국어 메시지 목록으로
  - `oom_retry(fn: Callable[[], T]) -> T` — `torch.cuda.OutOfMemoryError` 시 `empty_cache()` 후 1회 재시도, 재실패 시 `GpuOutOfMemory` raise
  - 래퍼 (모두 `oom_retry`로 개별 호출 감쌈):
    - `HFSegmenter` — `transformers.Sam2Model`/`Sam2Processor`, `input_boxes=[boxes]`, `multimask_output=False`, `post_process_masks`로 원본 크기 bool 마스크
    - `HFTextDetector` — `AutoModelForZeroShotObjectDetection`/`AutoProcessor`, text = `". ".join(p.lower() for p in phrases) + "."`, `post_process_grounded_object_detection`(설치된 버전의 인자명 확인: `threshold` 또는 `box_threshold`), 라벨 문자열을 가장 가까운 phrase로 매핑
    - `HFPromptGenerator` — `AutoModelForCausalLM` (Task 0 결정에 따라 fp16 또는 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=float16)`), chat template, 왼쪽 패딩 배치, `do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=96`, 요청별 `torch.manual_seed(req.seed)`로 배치 대신 순차 생성(결정성 우선)
    - `HFInpainter` — `AutoPipelineForInpainting.from_pretrained(id, torch_dtype=float16, variant="fp16")`(variant 없으면 재시도), `enable_model_cpu_offload()`, `vae.enable_slicing()`, `vae.enable_tiling()`; `native_px = 1024` if 파이프라인 클래스 이름에 `"XL"` 포함 else `512`; mask는 L 모드 255=인페인팅; `generator=torch.Generator("cuda").manual_seed(seed)`; `width/height`는 입력 크기
    - `HFValidator` — `Qwen2_5_VLForConditionalGeneration`/`AutoProcessor` (Task 0 결정 정밀도), `max_new_tokens=128`, greedy
- `pytest.ini`:
  ```ini
  [pytest]
  markers =
      gpu: requires CUDA and downloaded models
  addopts = -m "not gpu"
  ```

- [ ] **Step 1: Write the GPU smoke test**

```python
pytestmark = pytest.mark.gpu

def test_full_pipeline_one_image_on_gpu(tmp_path):
    s = _settings(tmp_path, n_images=1, num_variants=1, max_attempts=1, size=(512, 384),
                  keep_items=[KeepItem("floor", "color", hsv_lower=[0, 0, 0], hsv_upper=[179, 255, 80])])
    torch.cuda.reset_peak_memory_stats()
    BackgroundSwapWorkflow(s, HFModelFactory(s, ["box"]), NullProgress(), ["box"]).run()
    rec = read_jsonl(s.output_dir/"Manifest.jsonl")[-1]
    assert rec["status"] in ("accepted", "rejected") and (rec["status"] == "accepted" or rec["reasons"])
    assert torch.cuda.max_memory_allocated() < 8 * 1024**3
```

- [ ] **Step 2: Implement `models.py`, `pytest.ini`**

- [ ] **Step 3: Run default suite (GPU 제외 확인)**

Run: `python3 -m pytest -v`
Expected: PASS, `test_bg_gpu_smoke.py` deselected

- [ ] **Step 4: Run GPU smoke**

Run: `python3 -m pytest -m gpu -v -s`
Expected: PASS (첫 실행은 모델 다운로드로 오래 걸림)

- [ ] **Step 5: Run real CLI preview on the user's dataset**

사용자에게 train 경로와 keep 항목을 받아 `run_settings.json` 작성 후:
Run: `python3 -m background_swap preview --settings <path> --n 3`
Expected: 마지막 줄 `{"event": "done", "preview_dir": ...}`, `_preview/Accepted|Rejected`에 결과. 결과 이미지 3장과 차이 이미지를 사용자와 함께 확인.

- [ ] **Step 6: Commit**

```bash
git add background_swap/models.py pytest.ini tests/test_bg_gpu_smoke.py
git commit -m "feat(background_swap): add HF model wrappers and GPU smoke test"
```

---

### Task 13: GUI — 설정 패널, keep 항목 편집, 마스크 미리보기

**Files:**
- Create: `augmentor/gui/background_tab.py`
- Create: `augmentor/gui/background_state.py` (UI ↔ 설정 변환, 상태 파일 — Tk 의존 없음)
- Modify: `augmentor/gui/app.py` (탭 추가)
- Test: `augmentor/tests/test_bg_gui_state.py`

**Interfaces:**
- Consumes: `settings.*`, `classes.find_class_names_file`, `inputs.collect_items/load_image_bgr`, `color_region.hsv_range_from_click/color_mask`, `composite.mask_overlay/draw_boxes`, `masks.object_core_mask/object_protect_mask` (SAM 없이 사각형으로 즉시 미리보기), `core.label.read_labels`.
- Produces (`background_state.py`):
  - `UI_STATE_PATH = Path.home()/".config"/"dataset_helper"/"background_swap_ui.json"`
  - `load_ui_state(path: Path = UI_STATE_PATH) -> BackgroundSwapSettings` — 없거나 손상되면 기본 설정
  - `save_ui_state(s, path: Path = UI_STATE_PATH) -> None` — 예외는 삼킴
  - `parse_axis_values(text: str) -> list[str]` — `;` 또는 줄바꿈 구분, 공백 제거, 빈 값 제외
  - `format_axis_values(values: list[str]) -> str` — `"; ".join`
  - `quick_mask_preview(img_bgr, boxes: list[BBox], settings) -> np.ndarray` — SAM 없이 `object_core_mask(…, [None]*len, small_box_px)` + `object_protect_mask` + color 항목만 `color_mask` 합집합 → `mask_overlay` (text 항목은 Preview 실행 결과로만 표시)
- Produces (`background_tab.py`): `class BackgroundTab(ttk.Frame)` — `AugmentTab`과 같은 좌(스크롤 설정, minsize 280)/우(미리보기) 레이아웃
  - 설정 그룹(LabelFrame): **데이터**(Images/Labels/Classes/Output Entry+Browse, Frame stride Spinbox) · **유지할 배경**(항목 Treeview: 이름/방식/필수, [추가] [편집] [삭제]) · **생성**(Variants 1–20, Max attempts 1–10, Seed, Use LLM, Use VLM) · **장면 축**(축마다 Label + Entry(`format_axis_values`), [축 추가] [축 삭제]) · **고급**(접기 토글, 기본 접힘: resolution_mode Combobox, max/manual px, small_box_px, dilate_px, shadow_extend_ratio, feather_px, min_inpaint_ratio, min_change, 모델 ID 5개)
  - Images 선택 시 `find_class_names_file`로 Classes 자동 채움(비어 있을 때만)
  - keep 항목 편집 다이얼로그(`tk.Toplevel`): 이름, 방식 라디오(color/text), 필수 체크; color → "미리보기에서 클릭" 버튼(누르면 다음 미리보기 클릭 1회를 `hsv_range_from_click`에 사용, 원본 이미지 좌표로 역변환), 색 견본 Label, h/s/v tol Spinbox, largest_only, convex_hull, min_area_ratio; text → 프롬프트 Entry
  - 우측: 2×2 패널(원본+bbox / 마스크 오버레이 / 생성 결과 / 차이) — 이 Task에서는 위 두 개만 채우고 아래 두 개는 "Preview 실행 후 표시" 문구. Prev/Next로 원본 순회, 설정 변경 시 `quick_mask_preview` 갱신(150ms `after` 디바운스)
  - `build_settings() -> BackgroundSwapSettings`, `apply_settings(s) -> None` — 위젯 ↔ 설정; 탭 생성 시 `load_ui_state` 적용, `<Destroy>` 시 `save_ui_state`
  - `app.py`: `notebook.add(bg_frame, text="  🖼 Background Swap  ")`

- [ ] **Step 1: Write the failing tests (Tk 없는 부분)**

```python
def test_ui_state_roundtrip_and_corrupt(tmp_path):
    p = tmp_path/"ui.json"; s = BackgroundSwapSettings(num_variants=7)
    save_ui_state(s, p); assert load_ui_state(p).num_variants == 7
    p.write_text("{broken"); assert load_ui_state(p) == BackgroundSwapSettings()

def test_save_ui_state_swallows_errors(tmp_path):
    (tmp_path/"file.txt").write_text("x")
    save_ui_state(BackgroundSwapSettings(), tmp_path/"file.txt"/"x.json")  # 부모가 파일 → 예외 없이 통과

def test_parse_axis_values():
    assert parse_axis_values(" gym ; hall\n\n arena ;") == ["gym", "hall", "arena"]
    assert format_axis_values(["gym", "hall"]) == "gym; hall"

def test_quick_mask_preview_shape_and_colors():
    img = np.full((60, 80, 3), 128, np.uint8)
    out = quick_mask_preview(img, [BBox(0, 0.5, 0.5, 0.25, 0.25)], BackgroundSwapSettings())
    assert out.shape == img.shape and tuple(out[30, 40]) != (128, 128, 128)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_gui_state.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `background_state.py`, then `background_tab.py` and `app.py` 탭 등록**

- [ ] **Step 4: Run tests + launch check**

Run: `python3 -m pytest -v && python3 -c "import sys; import gui.background_tab; assert 'torch' not in sys.modules"`
Expected: PASS, 에러 없음 (GUI 모듈이 torch를 끌어오지 않음)

- [ ] **Step 5: Manual check (사용자와 함께)**

`python3 main.py` → Background Swap 탭: 폴더 선택 시 원본·마스크 오버레이 표시, Classes 자동 채움, color 항목 추가 후 필드 클릭 → 초록 오버레이, 고급 접기/펼치기, 앱 재시작 시 값 복원.

- [ ] **Step 6: Commit**

```bash
git add gui/background_tab.py gui/background_state.py gui/app.py tests/test_bg_gui_state.py
git commit -m "feat(gui): add Background Swap tab settings and mask preview"
```

---

### Task 14: GUI — 실행/중지, 진행, 결과 4패널, Report, Export

**Files:**
- Modify: `augmentor/gui/background_tab.py`
- Create: `augmentor/gui/worker_process.py` (서브프로세스 + JSON 라인 파서 — Tk 의존 없음)
- Test: `augmentor/tests/test_bg_worker_process.py`

**Interfaces:**
- Consumes: Task 11 CLI 이벤트/종료 코드, `settings.save_settings/load_settings/generation_diff/validate_settings`, `inputs.looks_non_train`, `store.read_jsonl/reset_output_dir`, `composite.diff_image/mask_overlay`, `store.read_mask`.
- Produces (`worker_process.py`):
  - `AUGMENTOR_DIR = Path(__file__).resolve().parents[1]`
  - `parse_event(line: str) -> dict | None` — JSON 객체가 아니면 `None`
  - `class WorkerProcess(args: list[str], on_event: Callable[[dict], None], on_exit: Callable[[int, str], None])` — `start()`: `subprocess.Popen([sys.executable, "-m", "background_swap", *args], cwd=AUGMENTOR_DIR, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)`, 읽기 스레드가 줄마다 `parse_event` → `on_event`, 종료 시 `on_exit(returncode, stderr 마지막 비어있지 않은 줄)`; `stop()`: `terminate()` → 5초 대기 → `kill()`; `running` 프로퍼티
- Produces (`background_tab.py` 추가):
  - 버튼 행: `Preview N=3`, `▶ Run`, `■ Stop`, `Report`, `Export`. 실행 중에는 Run/Preview/Export 비활성, Stop만 활성
  - Run 전 처리: `validate_settings` 오류 → `messagebox.showerror`; `looks_non_train(images_dir)` → `askyesno` 확인 시 `allow_non_train=True`; `output_dir/run_settings.json`이 있고 `generation_diff` 비어있지 않으면 3지선다 다이얼로그(**새 출력 폴더** → `askdirectory` 후 진행 / **덮어쓰기** → `reset_output_dir` 후 진행 / **취소**); 그다음 `save_settings(s, output_dir/"run_settings.json")` → `WorkerProcess(["run", "--settings", path])`
  - Preview: 같은 설정 저장 후 `["preview", "--settings", path, "--n", "3"]`, 완료 이벤트의 `preview_dir` 결과를 표시
  - 이벤트 처리(`after(0, ...)`로 메인 스레드): `stage` → 진행바(`done/total`)와 "단계: {한국어 단계명}" 라벨(`inputs`=입력, `masks`=마스크, `prompts`=프롬프트, `inpaint`=인페인팅, `validate`=검증, `report`=리포트); `warning` → 상태 라벨; `error` → `messagebox.showerror(message)`; 비정상 종료(코드 ≠ 0이고 error 이벤트 없음) → stderr 마지막 줄 표시; 종료 시 버튼 상태 복구
  - 결과 브라우징: 완료 후(또는 탭에서 output_dir 지정 시) Manifest의 accepted/rejected 레코드를 목록화, Prev/Next로 순회. 4패널 = 원본+bbox / `mask_overlay`(masks PNG) / 결과 이미지(Accepted 또는 Rejected 파일) / `diff_image(orig, result, protect)`. 하단에 상태, 사유 목록, `checks` 요약, 장면 축 값, 프롬프트
  - Report: `report.json`을 Toplevel 텍스트로 요약(상태/사유 집계, 장면 축 분포, 클래스별 recall + note 문구)
  - Export: 대상 images/labels 폴더 `askdirectory` 2회 → `looks_non_train`이면 확인 다이얼로그 → `WorkerProcess(["export", ...])` → 완료 이벤트의 copied/skipped_existing 메시지

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_event():
    assert parse_event('{"stage": "masks", "done": 1, "total": 2}\n') == {"stage": "masks", "done": 1, "total": 2}
    assert parse_event("Loading checkpoint shards: 50%") is None
    assert parse_event("[1, 2]") is None

def test_worker_process_reports_events_and_exit(tmp_path, monkeypatch):
    script = tmp_path/"fake_worker.py"
    script.write_text('import json,sys\nprint(json.dumps({"stage":"masks","done":0,"total":1}))\n'
                      'print("noise")\nprint(json.dumps({"event":"done"}))\nsys.stderr.write("last line\\n")\nsys.exit(0)\n')
    events, exits, done = [], [], threading.Event()
    wp = WorkerProcess([], events.append, lambda c, e: (exits.append((c, e)), done.set()),
                       command=[sys.executable, str(script)])
    wp.start(); assert done.wait(10)
    assert events == [{"stage": "masks", "done": 0, "total": 1}, {"event": "done"}]
    assert exits == [(0, "last line")]

def test_worker_process_stop_terminates(tmp_path):
    script = tmp_path/"sleep.py"; script.write_text("import time\ntime.sleep(60)\n")
    done = threading.Event()
    wp = WorkerProcess([], lambda e: None, lambda c, e: done.set(), command=[sys.executable, str(script)])
    wp.start(); wp.stop(); assert done.wait(10) and not wp.running
```

(`WorkerProcess`에 테스트용 키워드 인자 `command: list[str] | None = None` 추가 — 주어지면 `-m background_swap` 대신 사용.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_bg_worker_process.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `worker_process.py`, then 탭 실행/결과/Report/Export 기능**

- [ ] **Step 4: Run full suite**

Run: `python3 -m pytest -v`
Expected: PASS

- [ ] **Step 5: Manual checklist (사용자와 함께, 실제 데이터)**

1. Preview N=3 → 4패널 결과, 차이 패널의 보호 영역에 빨강 없음
2. Run → 단계별 진행바, 중간에 Stop → 다시 Run 시 이어서 진행(완료 이미지 재생성 안 됨)
3. seed 변경 후 Run → 3지선다 다이얼로그, 각 선택 동작
4. images 경로에 `val` 포함 시 확인 다이얼로그
5. Report 요약 표시, Export 후 데이터셋 폴더에 `_bgswap` 이미지/라벨, 재Export 시 skipped_existing 보고
6. requirements-bg 미설치 venv에서 Run → 설치 안내 에러

- [ ] **Step 6: Commit**

```bash
git add gui/background_tab.py gui/worker_process.py tests/test_bg_worker_process.py
git commit -m "feat(gui): run, monitor, review and export Background Swap jobs"
```

---

### Task 15: README 갱신

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README에 Background Swap 섹션 추가**

내용: 기능 한 문단, 설치(`pip install -r augmentor/requirements.txt -r augmentor/requirements-bg.txt`, HF 로그인 필요 모델 시 `hf auth login`), 사용 순서(탭 → 폴더 → 유지 항목 → Preview로 마스크 확인 → Run → Report → Export), 출력 폴더 구조, "train 이미지만 사용", 모델 라이선스 확인 문구(스펙 §9), 효과 평가 방법(스펙 §10) 요약.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document Background Swap usage"
```
