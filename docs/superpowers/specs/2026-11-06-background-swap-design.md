# Background Swap 설계 (LLM 기반 배경 교체 증강)

- 날짜: 2026-11-06
- 대상: `dataset_helper/augmentor`
- 상태: 설계 승인됨, 구현 플랜 작성 전

## 1. 목적과 배경

### 문제
현재 데이터셋은 **같은 실내 대회장 공간/조명**에서만 촬영되어 배경이 단조롭다. 그 결과 모델이 배경에 과적합되어 다른 대회장(새 환경)에서 일반화하지 못한다.

### 목표
라벨링된 이미지(YOLO bbox)에서 **라벨 객체와 "유지해야 하는 배경"(예: 경기장 필드 매트, 라인, 펜스)은 원본 픽셀 그대로 보존**하고, 나머지 배경(관중석, 벽, 천장, 조명 등 대회장마다 달라지는 부분)만 생성 모델로 교체하여 다양한 실내 대회장 배경의 학습 이미지를 늘린다.

### 성공 기준
1. 증강 이미지의 라벨이 원본 라벨을 그대로 재사용해도 정확하다 (객체·유지 영역 픽셀이 원본과 동일).
2. 교체된 배경이 그럴듯한 실내 대회장이며, 장면 축 조합에 따라 서로 충분히 다르다.
3. 배경에 **라벨 대상 클래스의 새 객체가 생기지 않는다** (생기면 라벨 누락 → false negative 학습).
4. RTX 4060 Laptop(VRAM 8GB)에서 OOM 없이 배치가 끝까지 돈다.
5. 모든 설정은 GUI에서 만들고, 사용자가 설정 파일을 직접 작성/편집하지 않는다.

### 참고
Gen-n-Val(`ws/src/Gen-n-Val`)의 "LLM 프롬프트 생성 → 이미지 생성 → VLM 검증 → Accepted/Rejected + Manifest, 재개 가능" 구조를 따르되, 생성 대상이 전경 객체가 아니라 **배경**이다.

## 2. 범위

### 포함
- bbox → 객체 마스크(SAM2), 텍스트 프롬프트 → 유지 배경 마스크(Grounding DINO + SAM2)
- LLM 장면 프롬프트 생성 (LLM 미사용 시 템플릿 폴백)
- SDXL 인페인팅 + 보호 영역 원본 재합성
- 규칙 검사 + 새 객체 검사 + VLM 검증
- 단계별 재개, Manifest, Accepted/Rejected 출력, 데이터셋으로 export
- GUI "Background Swap" 탭 (설정 입력, 미리보기, 실행/중지, 결과 탐색, export)

### 제외 (YAGNI)
- 이미지별 마스크 수동 브러시 편집 (마스크 PNG는 저장되므로 외부 툴로 수정 후 재실행은 가능)
- 클라우드 API 백엔드
- 멀티 GPU
- 세그멘테이션/폴리곤 라벨 포맷 (YOLO bbox만)
- 사용자 편집용 설정 파일 (YAML 등)

## 3. 아키텍처

### 패키지 구성
무거운 의존성(torch, diffusers, transformers, sam2 등)은 `augmentor/requirements-bg.txt`로 분리하고 **lazy import**한다. GPU/의존성이 없어도 기존 GUI와 기존 증강은 그대로 동작해야 한다. 기존 코드와 같이 `augmentor/`를 import 루트로 사용한다.

```
augmentor/
├── background_swap/
│   ├── __init__.py
│   ├── __main__.py     # python -m background_swap
│   ├── settings.py     # BackgroundSwapSettings 데이터클래스, 기본값, run_settings.json 직렬화
│   ├── domain.py       # ImageItem, MaskSet, Job, Attempt, ValidationResult
│   ├── classes.py      # 클래스 이름 파일 탐지/로드 (classes.txt, data.yaml names, obj.names)
│   ├── masks.py        # 객체/유지/보호/인페인팅 마스크 생성·합성 (numpy 로직 + Segmenter/Detector Protocol)
│   ├── prompts.py      # SceneSampler + PromptGenerator Protocol + LLM 구현 + 템플릿 구현
│   ├── inpaint.py      # Inpainter Protocol + SDXL-inpaint 구현
│   ├── composite.py    # 페더링 합성, 보호 영역 원본 복원, (옵션) 색 보정
│   ├── validate.py     # 규칙 검사, 새 객체 검사, VLM Validator Protocol + 구현, yes/no 파싱
│   ├── models.py       # 실제 HF 모델 래퍼, 단계 종료 시 GPU 해제
│   ├── workflow.py     # 단계 오케스트레이션, 재개, Manifest, export
│   ├── progress.py     # stdout JSON 진행 라인 출력
│   └── cli.py          # run / preview / export 서브커맨드 (run_settings.json 입력)
├── gui/background_tab.py
└── requirements-bg.txt
```

모든 모델 의존 컴포넌트는 `Protocol`로 정의하여(`Segmenter`, `TextDetector`, `PromptGenerator`, `Inpainter`, `ImageValidator`) 테스트에서 가짜 구현으로 교체한다.

### 8GB VRAM 전략: 단계별 일괄 처리
이미지마다 모델을 바꿔 올리지 않고, **한 단계에서 한 모델 계열만 GPU에 올려 전체 이미지를 처리한 뒤 해제**(`del` + `gc.collect()` + `torch.cuda.empty_cache()`)한다. 각 단계 산출물은 디스크에 저장되어 단계 단위 재개가 가능하다.

```
Stage 1 Masks     (Grounding DINO + SAM2)  → masks/{stem}_object.png, _keep.png, _protect.png
Stage 2 Prompts   (LLM)                    → jobs.jsonl
Stage 3 Inpaint   (SDXL-inpaint)           → generated/{stem}_bgswap{n}_a{k}.png  (재합성 완료본)
Stage 4 Validate  (규칙 + GDINO + VLM)     → Accepted/ | Rejected/ + Manifest.jsonl
Export                                     → 데이터셋 images/labels 폴더로 복사
```

재시도(거부된 변형)는 Stage 3–4를 거부된 변형에 대해서만 다시 반복한다(라운드 단위, 최대 `max_attempts` 라운드).

### 출력 폴더 구조
```
{output_dir}/
├── run_settings.json      # GUI가 자동 기록하는 실행 설정 (사용자 편집용 아님)
├── masks/
├── jobs.jsonl             # {stem, variant, attempt, seed, scene, prompt, negative_prompt}
├── generated/
├── Accepted/
│   ├── images/{stem}_bgswap{n}.jpg
│   ├── labels/{stem}_bgswap{n}.txt     # 원본 라벨 복사
│   └── previews/{stem}_bgswap{n}.jpg   # bbox 표시본 (검토용)
├── Rejected/
│   └── {stem}_bgswap{n}_a{k}.jpg
└── Manifest.jsonl         # 시도마다 1줄: stem, variant, attempt, seed, prompt, status, reasons, scores
```

## 4. 단계별 동작

### Stage 1 — 마스크
**객체 마스크** (bbox마다, 원본 해상도)
- bbox 짧은 변 ≥ `small_box_px`(기본 32px): SAM2 박스 프롬프트 → 가장 큰 연결 성분. SAM 마스크 면적이 bbox 면적의 20% 미만이면 실패로 보고 사각형 폴백.
- bbox 짧은 변 < `small_box_px`: bbox를 10% 여유로 확장한 사각형.
- 모든 객체 마스크 합집합 후 `dilate_px`(기본 4px) 팽창.

**유지 배경 마스크**
- `keep_prompts` 각각을 Grounding DINO(box 0.35 / text 0.25)로 검출 → 박스마다 SAM2 마스크 → 합집합.
- 어떤 프롬프트도 검출되지 않으면 빈 마스크로 진행하고 `keep_not_found` 경고를 Manifest에 기록.

**합성**
- `protect = object ∪ keep`, `inpaint = ~protect`
- `inpaint` 면적 비율 < `min_inpaint_ratio`(기본 0.15)이면 이미지 건너뜀(`skipped: small_inpaint_area`).

### Stage 2 — 장면 프롬프트
- `SceneSampler`: 시드(`seed`, stem, variant, attempt로 결정적 파생)로 장면 축마다 값 1개씩 선택.
- LLM(`Qwen/Qwen2.5-3B-Instruct`): 선택된 축 값 + 유지 대상(`keep_prompts`)을 받아 SDXL용 영어 프롬프트 1문장(≤60 단어)을 생성. 시스템 프롬프트로 **라벨 클래스 이름과 사람/로봇 등 객체 언급 금지**, 유지 대상을 "surrounding ..." 형태로 맥락에만 반영.
- 생성 결과에 클래스 이름이 포함되면 해당 단어를 제거(후처리).
- `negative_prompt`: 클래스 이름 전체 + 공통 품질 네거티브(blurry, distorted, text, watermark 등).
- `use_llm = false`면 템플릿: `"{venue}, {lighting}, {floor}, {context}, {camera}, surrounding {keep_prompts}, photorealistic"`.

### Stage 3 — 인페인팅 + 재합성
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, fp16, `enable_model_cpu_offload()`.
- 긴 변 1024 기준으로 리사이즈(8의 배수), strength 0.99, steps 30, guidance 7.0 → 원본 해상도로 복원.
- 합성: `w = feather(protect)` (가우시안 페더, 경계에서만 0~1) → `out = w·orig + (1−w)·gen`. 단, **팽창 전 객체 마스크와 유지 마스크 내부는 w=1로 강제**하여 원본 픽셀과 정확히 동일.
- 색 보정 옵션(`color_harmonize`, 기본 off): 생성 배경 통계에 맞춰 보호 영역을 약하게 보정. 켜면 규칙 검사 1의 "동일성"은 보정 전 기준으로 검사.

### Stage 4 — 검증
하나라도 실패하면 Rejected, 실패 사유를 모두 기록.
1. **규칙 검사**: (a) 출력 크기 == 원본 크기, (b) 보호 코어 영역 픽셀 == 원본, (c) 인페인팅 영역의 원본 대비 평균 절대 차이(0–255 픽셀값) ≥ `min_change`(기본 20) — 원본과 거의 같은 결과 배제.
2. **새 객체 검사**: 클래스 이름 목록으로 Grounding DINO 실행 → 기존 라벨 bbox 중 어느 것과도 IoU < 0.3인 검출(score ≥ 0.35)이 있으면 `new_object` 거부. 클래스 이름 파일이 없으면 이 검사는 건너뛰고 경고.
3. **VLM 검증**(`use_vlm = true`일 때, `Qwen/Qwen2.5-VL-3B-Instruct`): 결과 이미지(라벨 bbox 표시본)를 보여주고 "자연스러운 실내 대회장 사진인가, 표시된 객체 경계에 부자연스러운 흔적이 없는가"를 묻고 마지막 줄 `Answer: yes|no`를 파싱. 파싱 실패는 거부(`vlm_unparsable`).

### 수량과 재시도
- 이미지당 `num_variants`(기본 3, GUI 파라미터)개의 Accepted를 목표로 한다.
- 변형마다 거부 시 새 시드로 재시도, 최대 `max_attempts`(기본 3, GUI 파라미터). 소진 시 Manifest에 `exhausted`.

### Export
- `Accepted/images`, `Accepted/labels`를 GUI에서 지정한 데이터셋 images/labels 폴더로 복사. 동일 이름이 있으면 덮어쓰지 않고 건너뛰고 개수를 보고.
- 기존 `core.pipeline._collect_originals`에 `_bgswap` 제외 규칙을 추가하여 기존 증강이 export된 파일을 원본으로 오인하지 않게 한다.

## 5. 설정 (UI가 유일한 입력 지점)

`BackgroundSwapSettings` 데이터클래스가 모든 설정을 담는다. GUI가 값을 모아 생성하고, 실행 시 `{output_dir}/run_settings.json`으로 자동 기록하여 워커 서브프로세스에 전달한다. 사용자는 이 파일을 편집하지 않는다.

| 그룹 | 필드 | 기본값 | UI |
|---|---|---|---|
| 데이터 | `images_dir`, `labels_dir`, `output_dir` | — | Entry + Browse |
| 데이터 | `class_names_file` | 자동 탐지 (`classes.txt` → `data.yaml` → `obj.names`, labels/images 폴더와 그 상위) | Entry + Browse |
| 유지 배경 | `keep_prompts: list[str]` | 빈 목록 | 행 추가/삭제 리스트 |
| 생성 | `num_variants` | 3 | Spinbox (1–20) |
| 생성 | `max_attempts` | 3 | Spinbox (1–10) |
| 생성 | `seed` | 0 | Entry (정수) |
| 생성 | `use_llm`, `use_vlm` | true, true | Checkbutton |
| 장면 축 | `scene_axes: dict[str, list[str]]` | 아래 기본값 | 축별 항목 추가/삭제, 축 추가/삭제 |
| 고급 (접힘) | `small_box_px`, `dilate_px`, `min_inpaint_ratio`, `min_change`, `color_harmonize` | 32, 4, 0.15, 20, false | Spinbox/Entry/Check |
| 고급 (접힘) | 모델 ID 5개 (`segmenter`, `detector`, `llm`, `inpainter`, `vlm`) | 아래 기본값 | Entry |

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
- **우측 미리보기**: 원본 + 마스크 오버레이(빨강=객체, 초록=유지, 무색=교체 영역) | 생성 결과, Prev/Next, 상태(Accepted / Rejected + 사유).
- **진행바**: 현재 단계 이름 + done/total.
- **버튼**
  - `Preview N=3`: 무작위 3장(변형 1개씩)에 전체 파이프라인을 `{output_dir}/_preview/`로 실행 후 결과 표시. `_preview/`는 매 실행마다 비우고 다시 만들며, 재개/설정 비교 대상이 아니다.
  - `Run`: 설정 검증 → `run_settings.json` 기록 → `python -m background_swap run --settings ...` 서브프로세스 실행. stdout JSON 진행 라인(`{"stage": "...", "done": n, "total": m}`, `{"event": "done"}`, `{"event": "error", "message": "..."}`)을 스레드에서 읽고 `after()`로 UI 갱신.
  - `Stop`: 서브프로세스 종료. 재실행 시 이어서 진행.
  - `Export`: Accepted를 데이터셋 폴더로 복사, 결과 개수 메시지.
- GUI 프로세스는 torch를 import하지 않는다.

## 7. 에러 처리

| 상황 | 처리 |
|---|---|
| 의존성 미설치 | Run/Preview 시 워커가 import 실패를 `error` 이벤트로 보고 → GUI가 `pip install -r requirements-bg.txt` 안내 |
| 모델 접근 불가 | 실행 전 사전 점검(모델 ID별 접근 확인) 실패 시 시작하지 않고 목록 표시 |
| CUDA OOM | 해당 항목 1회 재시도(캐시 비움). 재실패 시 단계 중단, "고급 설정에서 더 작은 모델" 안내 |
| 이미지 읽기 실패 / 라벨 파일 없음 | 해당 이미지 건너뜀, Manifest `skipped` |
| keep 프롬프트 미검출 | 진행, `keep_not_found` 경고 |
| 인페인팅 영역 부족 | `skipped: small_inpaint_area` |
| 재시도 소진 | `exhausted` |
| 중단/크래시 | 모든 산출물은 임시 파일에 쓴 뒤 rename(원자적). 재실행 시 완료 산출물 건너뜀 |
| 기존 출력 폴더에 다른 설정으로 재개 | 생성에 영향 주는 필드 비교 → 다르면 GUI 다이얼로그: 새 출력 폴더 / 덮어쓰기 / 취소 |
| 워커 비정상 종료 | stderr 마지막 줄 표시, 버튼 상태 복구 |

## 8. 테스트

기존 `augmentor/tests/`에 pytest로 추가한다.

**GPU 없이 실행 (기본)**
- `test_bg_masks.py`: 작은 bbox 사각형 폴백, SAM 면적 부족 시 폴백, 팽창, `protect = object ∪ keep`, 인페인팅 비율, 건너뜀 판정
- `test_bg_composite.py`: 보호 코어 픽셀이 원본과 정확히 동일, 페더 경계 범위
- `test_bg_validate.py`: 규칙 검사, IoU 기반 새 객체 판정, VLM 응답 파싱(yes/no/파싱 실패)
- `test_bg_prompts.py`: 같은 시드 → 같은 장면, 클래스 이름 제거, 템플릿 폴백
- `test_bg_settings.py`: 데이터클래스 ↔ JSON 왕복, 설정 차이 비교, 클래스 파일 자동 탐지
- `test_bg_workflow.py`: 가짜 Protocol 구현으로 전체 단계 실행 → Accepted/Rejected 분류, 재시도/소진, 재개 시 건너뜀, Manifest 내용, export 이름 규칙과 충돌 건너뜀
- `test_pipeline.py` 추가: `_collect_originals`가 `_bgswap` 파일 제외

**GPU 스모크 (`@pytest.mark.gpu`, 기본 건너뜀)**
- 실제 모델로 512px 이미지 1장 전체 파이프라인 → 8GB에서 완료 및 Accepted 또는 사유 있는 Rejected

**GUI**: 수동 체크리스트 (탭 표시, 설정 입력/복원, Preview, Run/Stop/재개, Export).

## 9. 리스크

- **새 객체 생성**: 가장 큰 리스크. 네거티브 프롬프트 + GDINO 검사로 대응하나, 작은 객체 클래스는 누락 가능 → Accepted 미리보기 검토 권장.
- **조명 불일치**: 원본 객체/유지 영역 조명과 새 배경 조명이 어긋날 수 있음. 학습 다양성 목적상 허용하고, 필요 시 `color_harmonize` 사용.
- **keep 프롬프트 품질**: 텍스트 검출 품질에 의존 → Preview로 반드시 확인.
- **처리 시간**: 8GB 노트북 GPU에서 장당(변형당) 수십 초 예상. 대량 실행은 백그라운드로 장시간 실행 전제.
