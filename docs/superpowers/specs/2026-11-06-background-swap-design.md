# Background Swap 설계 (LLM 기반 배경 교체 증강)

- 날짜: 2026-11-06 (rev 2: 외부 리뷰 반영)
- 대상: `dataset_helper/augmentor`
- 상태: 설계 승인됨, 구현 플랜 작성 전

## 1. 목적과 배경

### 문제
현재 데이터셋은 **같은 실내 대회장 공간/조명**에서만 촬영되어 배경이 단조롭다. 그 결과 모델이 배경에 과적합되어 다른 대회장(새 환경)에서 일반화하지 못한다.

### 목표
라벨링된 이미지(YOLO bbox)에서 **라벨 객체와 "유지해야 하는 배경"(예: 경기장 필드/매트, 라인, 펜스, 골대)은 원본 픽셀 그대로 보존**하고, 나머지 배경(관중석, 벽, 천장, 조명 등 대회장마다 달라지는 부분)만 생성 모델로 교체하여 다양한 실내 대회장 배경의 학습 이미지를 늘린다.

### 성공 기준
1. 증강 이미지의 라벨이 원본 라벨을 그대로 재사용해도 정확하다 (객체·유지 영역 픽셀이 원본과 동일).
2. 교체된 배경이 그럴듯한 실내 대회장이며, 장면 축 조합에 따라 서로 충분히 다르다.
3. 배경에 **라벨 대상 클래스의 새 객체가 생기는 위험을 줄인다** (생기면 라벨 누락 → false negative 학습). 완전 보증은 불가능하며, 검사는 위험 감소 필터로 정의한다.
4. RTX 4060 Laptop(VRAM 8GB)에서 OOM 없이 배치가 끝까지 돈다.
5. 모든 설정은 GUI에서 만들고, 사용자가 설정 파일을 직접 작성/편집하지 않는다.
6. val/test 이미지에서 파생된 증강본이 학습 데이터에 섞이지 않는다.

### 참고
Gen-n-Val(`ws/src/Gen-n-Val`)의 "LLM 프롬프트 생성 → 이미지 생성 → VLM 검증 → Accepted/Rejected + Manifest, 재개 가능" 구조를 따르되, 생성 대상이 전경 객체가 아니라 **배경**이다.

## 2. 범위

### 포함
- 입력 선택: 프레임 간격 샘플링, train 전용 가드
- bbox → 객체 마스크(SAM2) + 그림자 영역 확장
- 유지 배경 항목: **색상 영역 방식**(HSV + 옵션 convex hull) 또는 **텍스트 검출 방식**(Grounding DINO + SAM2), 항목별 필수/선택
- LLM 장면 프롬프트 생성 (LLM 미사용 시 템플릿 폴백)
- 인페인팅(SDXL 기본, SD1.5 대체 가능, 작업 해상도는 원본 해상도 기반 자동 결정) + 보호 영역 원본 재합성
- 규칙 검사 + 새 객체 검사(GDINO + VLM) + VLM 구조화 검증
- 단계별 재개, Manifest, 실행 리포트(상태/사유 집계, 장면 축 통계, GDINO 클래스별 recall), Accepted/Rejected 출력, 데이터셋으로 export
- GUI "Background Swap" 탭 (설정 입력, 4패널 미리보기, 실행/중지, 결과 탐색, export)

### 제외 (YAGNI)
- 이미지별 마스크 수동 브러시 편집 (마스크 PNG는 저장되므로 외부 툴로 수정 후 재실행은 가능)
- CLIP 등 이미지 임베딩 기반 배경 중복 제거/다양성 제어 (v2 후보)
- 클라우드 API 백엔드, 멀티 GPU
- 세그멘테이션/폴리곤 라벨 포맷 (YOLO bbox만)
- 사용자 편집용 설정 파일 (YAML 등)
- 학습/평가 자동화 (§10 평가 방법은 수동 절차)

## 3. 아키텍처

### 패키지 구성
무거운 의존성(torch, diffusers, transformers 등)은 `augmentor/requirements-bg.txt`로 분리하고 **lazy import**한다. GPU/의존성이 없어도 기존 GUI와 기존 증강은 그대로 동작해야 한다. 기존 코드와 같이 `augmentor/`를 import 루트로 사용한다.

```
augmentor/
├── background_swap/
│   ├── __init__.py
│   ├── __main__.py     # python -m background_swap
│   ├── settings.py     # BackgroundSwapSettings / KeepItem 데이터클래스, 기본값, run_settings.json 직렬화
│   ├── domain.py       # ImageItem, MaskSet, Job, Attempt, ValidationResult
│   ├── inputs.py       # 원본 수집, 프레임 간격 샘플링, train 전용 가드
│   ├── classes.py      # 클래스 이름 파일 탐지/로드 (classes.txt, data.yaml names, obj.names)
│   ├── masks.py        # 객체 core/protect, 그림자 확장, 유지 마스크 합성, 인페인팅 마스크 (numpy)
│   ├── color_region.py # 클릭 픽셀 → HSV 범위, HSV 마스크, 모폴로지, 최대 성분, convex hull (numpy/cv2)
│   ├── prompts.py      # SceneSampler + PromptGenerator Protocol + LLM 구현 + 템플릿 구현
│   ├── inpaint.py      # Inpainter Protocol + diffusers AutoPipelineForInpainting 구현
│   ├── composite.py    # 알파 합성, 보호 영역 원본 복원, 차이 이미지
│   ├── validate.py     # 규칙 검사, 새 객체 판정, VLM Validator Protocol + 구현, JSON 파싱
│   ├── models.py       # 실제 HF 모델 래퍼(SAM2, GDINO, LLM, VLM), 단계 종료 시 GPU 해제
│   ├── report.py       # report.json 집계
│   ├── workflow.py     # 단계 오케스트레이션, 재개, Manifest, export
│   ├── progress.py     # stdout JSON 진행 라인 출력
│   └── cli.py          # run / preview / export 서브커맨드 (run_settings.json 입력)
├── gui/background_tab.py
└── requirements-bg.txt
```

모든 모델 의존 컴포넌트는 `Protocol`로 정의하여(`Segmenter`, `TextDetector`, `PromptGenerator`, `Inpainter`, `ImageValidator`) 테스트에서 가짜 구현으로 교체한다.

### 8GB VRAM 전략: 단계별 일괄 처리
이미지마다 모델을 바꿔 올리지 않고, **한 단계에서 한 모델 계열만 GPU에 올려 전체 이미지를 처리한 뒤 해제**(`del` + `gc.collect()` + `torch.cuda.empty_cache()`)한다. 서로 다른 단계의 모델은 동시에 GPU에 존재하지 않는다. 각 단계 산출물은 디스크에 저장되어 단계 단위 재개가 가능하다.

```
Stage 0 Inputs    (모델 없음)               → items.jsonl   (선택된 원본 목록)
Stage 1 Masks     (SAM2, GDINO)             → masks/{stem}_{object_core,object_protect,keep,inpaint}.png
                                              + GDINO 클래스별 recall (원본 기준)
Stage 2 Prompts   (LLM)                     → jobs.jsonl
Stage 3 Inpaint   (SDXL/SD1.5 inpaint)      → generated/{stem}_bgswap{n}_a{k}.png  (재합성 완료본)
Stage 4 Validate  (규칙 + GDINO + VLM)      → Accepted/ | Rejected/ + Manifest.jsonl
Report                                      → report.json
Export                                      → 데이터셋 images/labels 폴더로 복사
```

재시도(거부된 변형)는 Stage 3–4를 거부된 변형에 대해서만 다시 반복한다(라운드 단위, 최대 `max_attempts` 라운드). 재시도 프롬프트는 새 시드로 Stage 2 로직을 다시 호출하여 만든다(LLM을 재로딩해야 하므로 재시도 라운드 시작 시 LLM 단계 → 인페인팅 단계 → 검증 단계 순으로 돈다).

### 출력 폴더 구조
```
{output_dir}/
├── run_settings.json      # GUI가 자동 기록하는 실행 설정 (사용자 편집용 아님)
├── items.jsonl
├── masks/
├── jobs.jsonl             # {stem, variant, attempt, seed, scene, prompt, negative_prompt}
├── generated/
├── Accepted/
│   ├── images/{stem}_bgswap{n}.jpg
│   ├── labels/{stem}_bgswap{n}.txt     # 원본 라벨 복사
│   └── previews/{stem}_bgswap{n}.jpg   # bbox 표시본 (검토용)
├── Rejected/
│   └── {stem}_bgswap{n}_a{k}.jpg
├── Manifest.jsonl         # 시도마다 1줄: stem, variant, attempt, seed, scene, prompt, status, reasons, checks
└── report.json
```

## 4. 단계별 동작

### Stage 0 — 입력 선택
- 원본 수집: 기존 `_collect_originals` 규칙 + `_bgswap` 제외.
- **프레임 간격 샘플링** `frame_stride`(기본 1 = 전부): 파일명 정렬 순서에서 N장마다 1장 선택. 영상에서 추출한 연속 프레임의 중복 생성을 줄인다.
- **train 전용 가드**: `images_dir` 또는 export 대상 경로의 구성요소에 `val`, `valid`, `validation`, `test`가 있으면 GUI가 경고 다이얼로그로 명시적 확인을 요구한다 (CLI는 `run_settings.json`의 `allow_non_train: true` 없이는 거부).

### Stage 1 — 마스크
마스크는 모두 원본 해상도의 이진 마스크이며 다음 4종을 명확히 구분한다.

| 마스크 | 정의 | 용도 |
|---|---|---|
| `object_core` | 객체의 실제 영역 (SAM2 또는 사각형 폴백) | 라벨 객체 자체 |
| `object_protect` | `dilate(object_core, dilate_px)` ∪ 그림자 확장 영역 | 경계 링·그림자·모션 블러까지 원본 보존 |
| `keep` | 유지 배경 항목 마스크의 합집합 | 경기장 구조 보존 |
| `inpaint` | `~(object_protect ∪ keep)` | 생성 모델이 새로 그릴 영역 |

**object_core** (bbox마다)
- bbox 짧은 변 ≥ `small_box_px`(기본 32px): SAM2 박스 프롬프트 → 가장 큰 연결 성분. SAM 마스크 면적이 bbox 면적의 20% 미만이면 실패로 보고 사각형 폴백.
- bbox 짧은 변 < `small_box_px`: bbox를 10% 여유로 확장한 사각형.

**object_protect**
- `object_core`를 `dilate_px`(기본 6px) 팽창.
- 그림자 확장: 각 bbox 아래쪽으로 높이 `shadow_extend_ratio × bbox_h`(기본 0.15), 폭 = bbox 폭 × 1.2 의 사각형을 추가 (이미지 경계로 클립). 0이면 비활성.

**keep — 유지 배경 항목** (`KeepItem` 목록, 항목마다 방식 선택)
- 공통 필드: `name`, `method: "color" | "text"`, `required: bool`
- **color 방식** (필드/매트 등 넓은 단색 영역)
  - GUI 미리보기에서 영역을 클릭하면 클릭 지점 주변 5×5 픽셀의 HSV 중앙값 ± 허용치(`h_tol` 10, `s_tol` 60, `v_tol` 60 기본)로 `hsv_lower/hsv_upper`를 설정. 허용치는 항목별로 조정 가능. H는 OpenCV 0–179 범위이며 0/179 경계를 넘는 범위(빨강 계열)는 두 구간으로 처리.
  - 처리: HSV 범위 마스크 → 모폴로지 open/close(커널 5) → `largest_only`(기본 true)면 최대 연결 성분 → `convex_hull`(기본 true)면 볼록 껍질로 채움. convex hull을 쓰면 필드 내부의 라인·마킹·작은 물체도 함께 유지된다.
  - 성공 판정: 결과 면적이 이미지의 `min_area_ratio`(기본 0.05) 이상.
- **text 방식** (펜스, 골대 등 물체 형태)
  - `prompt` 텍스트로 Grounding DINO(box 0.35 / text 0.25) 검출 → 박스마다 SAM2 마스크 → 합집합.
  - 성공 판정: 검출 1개 이상.
- **필수 항목 실패 시** 이미지 전체를 건너뜀(`skipped: required_keep_missing:<name>`). **선택 항목 실패 시** 해당 항목 없이 진행하고 `keep_not_found:<name>` 경고 기록.

**건너뜀 조건**: `inpaint` 면적 비율 < `min_inpaint_ratio`(기본 0.15) → `skipped: small_inpaint_area`.

**GDINO 클래스별 recall (검사 신뢰도 측정)**: 클래스 이름 파일이 있으면 Stage 1에서(이미 GDINO가 로드된 상태) 원본 이미지에 클래스 이름으로 GDINO를 돌려, 클래스별로 라벨 bbox 중 IoU ≥ 0.5로 검출된 비율을 계산해 `report.json`에 기록한다. Stage 4 새 객체 검사가 클래스별로 얼마나 믿을 만한지의 지표다.

### Stage 2 — 장면 프롬프트
- `SceneSampler`: 시드(`seed`, stem, variant, attempt로 결정적 파생)로 장면 축마다 값 1개씩 선택. 선택 결과(축→값)는 `jobs.jsonl`에 기록.
- LLM(`Qwen/Qwen2.5-3B-Instruct`): 선택된 축 값 + 유지 항목 이름을 받아 SDXL용 영어 프롬프트 1문장(≤60 단어)을 생성. 시스템 프롬프트로 **라벨 클래스 이름 및 사람·로봇·공 등 셀 수 있는 객체 언급 금지**, 유지 대상은 "surrounding ..." 형태로 맥락에만 반영.
- 생성 결과에 클래스 이름이 포함되면 해당 단어를 제거(후처리).
- `negative_prompt`: 클래스 이름 전체 + 공통 품질 네거티브(blurry, distorted, text, watermark, extra objects 등).
- `use_llm = false`면 템플릿: `"{venue}, {lighting}, {surroundings_floor}, {context}, {camera}, surrounding {keep names}, photorealistic"`.

### Stage 3 — 인페인팅 + 재합성
- `diffusers.AutoPipelineForInpainting`으로 모델 ID에 맞는 파이프라인을 로드한다. 기본 `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, 대체 예: `stable-diffusion-v1-5/stable-diffusion-inpainting`. fp16, `enable_model_cpu_offload()`, VAE slicing/tiling 활성.
- **작업 해상도는 이미지마다 원본 해상도에서 자동 결정**한다 (`resolution_mode = "auto"`, 기본).
  1. 원본 크기(파일에서 읽음)에서 시작한다.
  2. 긴 변이 모델의 기본 해상도(`model_native_px`: SDXL 1024, SD1.5 512 — 파이프라인 종류로 자동 판정)보다 작으면 그 값까지 업스케일한다. 기본 해상도보다 작게 생성하면 품질이 크게 떨어지기 때문이다.
  3. 긴 변이 `max_resolution_px`(VRAM 상한, 기본값은 §11 실측으로 확정)보다 크면 그 값까지 다운스케일한다.
  4. 가로세로 비율을 유지하고 각 변을 8의 배수로 맞춘다.
  - 생성 후 **항상 원본 해상도로 복원**하므로 라벨 좌표와 보호 영역 합성은 원본 기준 그대로다.
  - `resolution_mode = "manual"`이면 긴 변을 `manual_resolution_px`로 고정한다 (고급 설정, 실험용).
- 생성 파라미터: strength 0.99, steps 30, guidance 7.0.
- 인페인팅 모델에 주는 마스크 = `inpaint`.
- 합성: 알파 `a`를 `object_protect ∪ keep` 내부는 1, 그 바깥으로 `feather_px`(기본 4px) 거리에 걸쳐 1→0으로 감소, 나머지 0으로 정의 → `out = a·orig + (1−a)·gen`. 따라서 **`object_protect ∪ keep` 내부 픽셀은 원본과 정확히 동일**하고, 페더는 인페인팅 영역 쪽 경계에만 생긴다.
- **차이 이미지**: `|out − orig|`의 그레이 맵. 보호 영역에서 0이 아닌 픽셀은 빨강으로 표시 (정상이라면 없음 — 마스크/합성 버그 탐지용). Accepted/Rejected 미리보기와 GUI에서 사용.

### Stage 4 — 검증
하나라도 실패하면 Rejected, 실패 사유를 모두 `reasons`에, 개별 결과를 `checks`에 기록.
1. **규칙 검사**: (a) 출력 크기 == 원본 크기, (b) `object_protect ∪ keep` 픽셀 == 원본, (c) `inpaint` 영역의 원본 대비 평균 절대 차이(0–255 픽셀값) ≥ `min_change`(기본 20) — 원본과 거의 같은 결과 배제.
2. **새 객체 검사 (위험 감소 필터)**: 클래스 이름으로 Grounding DINO 실행 → 기존 라벨 bbox 중 어느 것과도 IoU < 0.3이면서 검출 박스 면적의 50% 이상이 `inpaint` 영역에 있는 검출(score ≥ 0.35)이 있으면 `new_object:gdino`. 클래스 이름 파일이 없으면 건너뛰고 경고. 미검출이 "새 객체 없음"을 의미하지 않음을 리포트/GUI에 명시.
3. **VLM 구조화 검증**(`use_vlm = true`일 때, `Qwen/Qwen2.5-VL-3B-Instruct`): bbox 표시본을 입력하고 아래 JSON만 출력하도록 지시.
   ```json
   {"natural_indoor_venue": true, "boundary_artifacts": false, "new_objects": false, "reason": "..."}
   ```
   - `natural_indoor_venue == false` → `vlm_unnatural`
   - `boundary_artifacts == true` → `vlm_boundary_artifacts`
   - `new_objects == true` → `new_object:vlm` (2번 GDINO 결과와 OR)
   - JSON 파싱 실패 → `vlm_unparsable` (거부)
   - 유지 영역 훼손은 규칙 검사 1(b)로 결정적으로 보장되므로 VLM에 묻지 않는다. 조명 일관성은 3B 모델의 판단 신뢰도가 낮아 v1에서 제외.

### 수량과 재시도
- 이미지당 `num_variants`(기본 3, GUI 파라미터)개의 Accepted를 목표로 한다.
- 변형마다 거부 시 새 시드로 재시도, 최대 `max_attempts`(기본 3, GUI 파라미터). 소진 시 Manifest에 `exhausted`.

### Report — `report.json`
- 상태별 개수 (accepted / rejected / skipped / exhausted), 사유별 개수
- 장면 축별 값 사용 횟수 (전체 / Accepted 기준) — 특정 배경 유형 편중 확인용
- GDINO 클래스별 recall (Stage 1)
- 단계별 소요 시간, 이미지당 평균 생성 시간

### Export
- `Accepted/images`, `Accepted/labels`를 GUI에서 지정한 데이터셋 images/labels 폴더로 복사. 동일 이름이 있으면 덮어쓰지 않고 건너뛰고 개수를 보고. train 전용 가드(Stage 0)를 export 대상 경로에도 적용.
- 기존 `core.pipeline._collect_originals`에 `_bgswap` 제외 규칙을 추가하여 기존 증강이 export된 파일을 원본으로 오인하지 않게 한다.

## 5. 설정 (UI가 유일한 입력 지점)

`BackgroundSwapSettings` 데이터클래스가 모든 설정을 담는다. GUI가 값을 모아 생성하고, 실행 시 `{output_dir}/run_settings.json`으로 자동 기록하여 워커 서브프로세스에 전달한다. 사용자는 이 파일을 편집하지 않는다.

| 그룹 | 필드 | 기본값 | UI |
|---|---|---|---|
| 데이터 | `images_dir`, `labels_dir`, `output_dir` | — | Entry + Browse |
| 데이터 | `class_names_file` | 자동 탐지 (`classes.txt` → `data.yaml` → `obj.names`, labels/images 폴더와 그 상위) | Entry + Browse |
| 데이터 | `frame_stride` | 1 | Spinbox (1–100) |
| 유지 배경 | `keep_items: list[KeepItem]` | 빈 목록 | 항목 리스트 (추가/삭제/편집, 아래 참고) |
| 생성 | `num_variants` | 3 | Spinbox (1–20) |
| 생성 | `max_attempts` | 3 | Spinbox (1–10) |
| 생성 | `seed` | 0 | Entry (정수) |
| 생성 | `use_llm`, `use_vlm` | true, true | Checkbutton |
| 장면 축 | `scene_axes: dict[str, list[str]]` | 아래 기본값 | 축별 항목 추가/삭제, 축 추가/삭제 |
| 고급 (접힘) | `resolution_mode` | auto (원본 해상도 기반, §4 Stage 3) | Combobox (auto/manual) |
| 고급 (접힘) | `max_resolution_px` | §11 실측으로 확정 | Spinbox |
| 고급 (접힘) | `manual_resolution_px` | 1024 (manual일 때만 사용) | Spinbox |
| 고급 (접힘) | `small_box_px`, `dilate_px`, `shadow_extend_ratio`, `feather_px`, `min_inpaint_ratio`, `min_change` | 32, 6, 0.15, 4, 0.15, 20 | Spinbox/Entry |
| 고급 (접힘) | 모델 ID 5개 (`segmenter`, `detector`, `llm`, `inpainter`, `vlm`) | 아래 기본값 | Entry |

**KeepItem 편집 UI**: 이름, 방식(color/text 라디오), 필수 체크박스.
- color: "미리보기에서 클릭하여 색 선택" 버튼 → 클릭 시 HSV 범위 설정, 색 견본 표시, `h_tol/s_tol/v_tol`, `largest_only`, `convex_hull`, `min_area_ratio`.
- text: 프롬프트 Entry.
- 편집 즉시 현재 미리보기 이미지에 해당 항목 마스크를 오버레이 (color 방식은 GUI 프로세스에서 cv2로 즉시 계산 가능, text 방식은 Preview 실행 시 반영).

**장면 축 기본값**
- venue: school gymnasium, convention center exhibition hall, university auditorium, warehouse-style arena, sports hall
- lighting: cool fluorescent lights, warm LED lighting, stage spotlights, dim indoor lighting, daylight from high windows
- surroundings_floor: grey carpet, polished concrete, wooden court floor, vinyl floor mats
- context: spectators in bleachers, sponsor banners on walls, equipment tables and laptops, empty hall
- camera: smartphone photo with noise, clean DSLR photo, slight motion blur

**모델 기본값**
- segmenter: `facebook/sam2.1-hiera-small`
- detector: `IDEA-Research/grounding-dino-tiny`
- llm: `Qwen/Qwen2.5-3B-Instruct`
- inpainter: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- vlm: `Qwen/Qwen2.5-VL-3B-Instruct`

마지막 UI 상태는 `~/.config/dataset_helper/background_swap_ui.json`에 자동 저장/복원한다 (편의 기능, 실패해도 무시).

## 6. GUI — `gui/background_tab.py`

`gui/app.py`의 Notebook에 "Background Swap" 탭을 추가한다. 레이아웃과 스타일은 `augment_tab.py`(좌측 스크롤 설정 패널 + 우측 미리보기)를 따른다.

- **좌측 설정 패널**: 5개 그룹(데이터 / 유지할 배경 / 생성 / 장면 축 / 고급).
- **우측 미리보기 — 4패널 (2×2)**
  1. 원본 (+ 라벨 bbox)
  2. 마스크 오버레이: 빨강=`object_core`, 주황=`object_protect` 링/그림자, 초록=`keep`, 무색=`inpaint`
  3. 생성 결과
  4. 차이 이미지 (보호 영역 차이는 빨강)
  - 하단: Prev/Next, 상태(Accepted / Rejected + 사유 목록, 각 검사 결과), 사용된 장면 축 값과 프롬프트
- **진행바**: 현재 단계 이름 + done/total.
- **버튼**
  - `Preview N=3`: 무작위 3장(변형 1개씩)에 전체 파이프라인을 `{output_dir}/_preview/`로 실행 후 결과 표시. `_preview/`는 매 실행마다 비우고 다시 만들며, 재개/설정 비교 대상이 아니다.
  - `Run`: 설정 검증(+ train 전용 가드) → `run_settings.json` 기록 → `python -m background_swap run --settings ...` 서브프로세스 실행. stdout JSON 진행 라인(`{"stage": "...", "done": n, "total": m}`, `{"event": "done"}`, `{"event": "error", "message": "..."}`)을 스레드에서 읽고 `after()`로 UI 갱신.
  - `Stop`: 서브프로세스 종료. 재실행 시 이어서 진행.
  - `Report`: `report.json` 요약 표시 (상태/사유 집계, 장면 축 분포, GDINO recall + "미검출 ≠ 새 객체 없음" 안내).
  - `Export`: Accepted를 데이터셋 폴더로 복사, 결과 개수 메시지.
- GUI 프로세스는 torch를 import하지 않는다 (cv2/numpy/PIL만).

## 7. 에러 처리

| 상황 | 처리 |
|---|---|
| 의존성 미설치 | Run/Preview 시 워커가 import 실패를 `error` 이벤트로 보고 → GUI가 `pip install -r requirements-bg.txt` 안내 |
| 모델 접근 불가 | 실행 전 사전 점검(모델 ID별 접근 확인) 실패 시 시작하지 않고 목록 표시 |
| CUDA OOM | 해당 항목 1회 재시도(캐시 비움). 재실패 시 단계 중단, "고급 설정에서 해상도를 낮추거나 더 작은 모델" 안내 |
| 이미지 읽기 실패 / 라벨 파일 없음 | 해당 이미지 건너뜀, Manifest `skipped` |
| 지원하지 않는 라벨 형식 (5필드가 아닌 줄, 정수가 아닌 클래스 등) | 이미지 건너뜀, `skipped: unsupported_label_format` |
| 필수 유지 항목 실패 | 이미지 건너뜀, `skipped: required_keep_missing:<name>` |
| 선택 유지 항목 실패 | 진행, `keep_not_found:<name>` 경고 |
| 인페인팅 영역 부족 | `skipped: small_inpaint_area` |
| 재시도 소진 | `exhausted` |
| val/test 경로 감지 | GUI 확인 다이얼로그, CLI는 `allow_non_train` 없으면 거부 |
| 중단/크래시 | 모든 산출물은 임시 파일에 쓴 뒤 rename(원자적). 재실행 시 완료 산출물 건너뜀 |
| 기존 출력 폴더에 다른 설정으로 재개 | 생성에 영향 주는 필드 비교 → 다르면 GUI 다이얼로그: 새 출력 폴더 / 덮어쓰기 / 취소 |
| 워커 비정상 종료 | stderr 마지막 줄 표시, 버튼 상태 복구 |

## 8. 테스트

기존 `augmentor/tests/`에 pytest로 추가한다.

**GPU 없이 실행 (기본)**
- `test_bg_inputs.py`: 프레임 간격 샘플링, `_bgswap` 제외, val/test 경로 감지
- `test_bg_masks.py`: 작은 bbox 사각형 폴백, SAM 면적 부족 시 폴백, 팽창, 그림자 확장(경계 클립 포함), `inpaint = ~(object_protect ∪ keep)`, 필수/선택 유지 실패 처리, 인페인팅 비율 건너뜀
- `test_bg_color_region.py`: 클릭 → HSV 범위(빨강 계열 H 경계 포함), 최대 성분, convex hull이 내부 라인을 채우는지, 최소 면적 판정
- `test_bg_composite.py`: 보호 영역 픽셀이 원본과 정확히 동일, 페더가 인페인팅 쪽에만 존재, 차이 이미지에서 보호 영역 0
- `test_bg_validate.py`: 규칙 검사, IoU + inpaint 영역 비율 기반 새 객체 판정, VLM JSON 파싱(정상/필드 누락/파싱 실패), GDINO/VLM OR 결합
- `test_bg_prompts.py`: 같은 시드 → 같은 장면, 클래스 이름 제거, 템플릿 폴백
- `test_bg_settings.py`: 데이터클래스(KeepItem 포함) ↔ JSON 왕복, 설정 차이 비교, 클래스 파일 자동 탐지
- `test_bg_report.py`: 상태/사유 집계, 장면 축 통계, recall 계산
- `test_bg_workflow.py`: 가짜 Protocol 구현으로 전체 단계 실행 → Accepted/Rejected 분류, 재시도/소진, 재개 시 건너뜀, Manifest 내용, export 이름 규칙과 충돌 건너뜀
- `test_pipeline.py` 추가: `_collect_originals`가 `_bgswap` 파일 제외

**GPU 스모크 (`@pytest.mark.gpu`, 기본 건너뜀)**
- 실제 모델로 512px 이미지 1장 전체 파이프라인 → 8GB에서 완료 및 Accepted 또는 사유 있는 Rejected

**GUI**: 수동 체크리스트 (탭 표시, 설정 입력/복원, color 항목 클릭 선택과 즉시 오버레이, Preview 4패널, Run/Stop/재개, Report, Export).

## 9. 리스크

- **새 객체 생성**: 가장 큰 리스크. 네거티브 프롬프트 + GDINO + VLM으로 위험을 줄일 뿐 보증하지 않는다. 작은 객체 클래스는 GDINO recall이 낮을 수 있으므로 report의 클래스별 recall을 확인하고 Accepted 미리보기 표본 검토를 권장.
- **마스크 품질**: 객체와 유지 구조물이 인접한 장면에서 경계 분리 오류가 라벨 품질을 떨어뜨릴 수 있다. 소량 Preview로 마스크를 먼저 검증한 뒤 대량 실행한다.
- **조명 불일치**: 원본 객체/유지 영역 조명과 새 배경 조명이 어긋날 수 있음. 학습 다양성 목적상 v1에서는 허용.
- **유지 항목 품질**: color 방식은 배경에 비슷한 색이 있으면 과검출 가능(최대 성분 옵션으로 완화), text 방식은 검출기 품질에 의존 → Preview로 반드시 확인.
- **처리 시간**: 8GB 노트북 GPU에서 변형당 수십 초 예상. `frame_stride`로 입력 수를 줄이고, 대량 실행은 장시간 백그라운드 실행 전제.
- **모델 라이선스**: 기본 LLM/VLM(Qwen2.5-3B, Qwen2.5-VL-3B)은 비상업 연구용 라이선스로 알려져 있다 — 사용 전 각 모델 카드에서 확인. 상업적 용도라면 허용 라이선스 모델로 모델 ID를 교체한다. SDXL inpainting은 CreativeML OpenRAIL++-M.

## 10. 효과 평가 방법 (수동 절차, 코드 범위 밖)

1. 학습에 쓰지 않는 **다른 환경(다른 대회장/공간/조명)에서 촬영한 소량의 테스트셋**을 별도로 확보·라벨링한다.
2. 같은 학습 설정으로 (a) 원본만, (b) 원본 + Background Swap 증강으로 학습한다.
3. 기존 val셋과 새 환경 테스트셋 모두에서 mAP를 비교한다. 목표: 새 환경 테스트셋에서 (b)가 개선되고 기존 val셋에서 크게 떨어지지 않는다.
4. 합성 비율(원본:증강 = 1:0.5, 1:1, 1:2)을 변수로 비교한다.

## 11. 구현 순서 원칙

1. **VRAM 실측 스파이크** (버릴 코드): 실제 데이터셋 이미지 크기 분포를 확인하고, 그 비율로 긴 변 512/768/1024/1280에서 SDXL inpaint의 피크 VRAM·장당 시간을 측정, SD1.5 비교 → `max_resolution_px` 기본값 확정.
2. 설정/입력/마스크/color region + GUI 4패널 미리보기(마스크까지) — **생성 전에 소량 이미지로 마스크 정확도를 먼저 검증**.
3. 프롬프트 → 인페인팅·합성 → 검증 → 재개/Manifest/Report → Export.
