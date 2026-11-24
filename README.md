# dataset_helper

데이터셋을 만드는 GUI 도구(augmentor)다. 탭 세 개로 이루어져 있다.

- **Video Tool**: 동영상(.webm 등)을 .mp4로 변환하고, 프레임을 RGB 또는 그레이스케일 이미지로 추출한다.
- **Augmentation**: YOLO 라벨이 있는 이미지를 뒤집기, 회전, 자르기, 크기 조절, 기울이기, 밝기, 블러, 노이즈, CutMix로 증강한다 (라벨도 함께 변환).
- **Background Swap**: 라벨 객체는 그대로 두고 배경만 새로 그려 증강한다 (아래 설명).

실행과 설치는 저장소 루트에서 `make`로 한다. `make`만 치면 전체 명령 목록이 나온다.

```bash
cd ~/ws/src/dataset_helper
make deps       # 처음 한 번: 시스템 패키지 설치 (sudo 필요, Ubuntu/Debian)
make venv       # Python 가상환경 + Large 의존성 (torch는 별도, 아래 "설치" 참고)
make venv-small # (작은 GPU용) Small 의존성만 설치
make augmentor  # GUI 실행
make test       # 테스트 (GPU 테스트 제외)
```

창 오른쪽 위 **`📁 기본 폴더`**로 자주 쓰는 폴더를 정해 두면 모든 Browse 창이 그 폴더에서 열린다. 각 탭에서 마지막으로 고른 경로는 앱을 껐다 켜도 다시 채워진다 (`~/.config/dataset_helper/app_prefs.json`).

Augmentation 탭과 Background Swap 탭 오른쪽 아래의 **`ⓘ Info`** 버튼을 누르면 화면의 모든 항목이 무엇을 뜻하는지 한국어로 설명한 창이 뜬다. 처음 쓴다면 먼저 열어 보자.

---

## 사양

### 실행 환경

| 항목 | 요구 사항 | 확인한 환경 |
|---|---|---|
| OS | Linux (Ubuntu/Debian). `make deps`가 `apt`를 쓴다 | Ubuntu 24.04 |
| Python | **3.12 이상** (3.11 이하에서는 background_swap 모듈 import가 실패한다) | 3.12.3 |
| 시스템 패키지 | `python3-venv`, `python3-tk`, `ffmpeg` (`make deps`로 설치) | |
| GPU | Background Swap만 **NVIDIA CUDA GPU** 필요 (장치가 `cuda`로 고정, CPU 실행 없음). 다른 두 탭은 GPU 없이 동작 | RTX 4060 Laptop (VRAM 8GB) |
| torch | Background Swap만 필요. requirements에 없으므로 그 PC의 CUDA 버전에 맞게 따로 설치 | torch 2.13.0 + cu130 |
| RAM | 최소치는 재지 않았다. VRAM이 작으면 모델 일부를 CPU 메모리로 내리므로 RAM을 더 쓴다 | 62GB |
| 디스크 | Large 모델 첫 다운로드 합계 약 24GB (`~/.cache/huggingface`) + 출력 이미지 | |

### 탭별 요구 사항

| 탭 | GPU | 필요한 것 |
|---|---|---|
| Video Tool | 불필요 | `ffmpeg`, `requirements.txt` |
| Augmentation | 불필요 | `requirements.txt` (opencv-python, numpy, Pillow, pyyaml) |
| Background Swap | NVIDIA CUDA | torch + `requirements-bg-large.txt` 또는 `requirements-bg-small.txt`, 첫 실행 시 모델 다운로드 |

### Background Swap: GPU별 권장 모델

모델 구성(Large / Small)은 아래 [설치](#설치)의 표를 본다. 단계마다 모델을 한 종류만 GPU에 올리므로, 필요한 VRAM은 가장 큰 단계 하나로 정해진다.

| VRAM | 권장 | 비고 |
|---|---|---|
| 7GB 이상 (예: 8GB) | **Large** (기본) | 속도가 필요하면 Small 대신 **빠른 모드**를 켠다 |
| 7GB 미만 (4~6GB급) | Small | Large도 돌지만 SDXL이 순차 오프로드로 바뀌어 매우 느리다. Small은 4GB급 GPU를 목표로 만들었지만, 실행은 8GB PC에서만 확인했다 |
| CUDA GPU 없음 | 사용 불가 | Video Tool과 Augmentation 탭만 쓸 수 있다 |

Small의 인페인팅(SD1.5)은 512px로 그린 뒤 원본 크기로 늘리므로 새 배경이 원본보다 흐리다. 또 VLM 검수가 꺼져 있어 잘못 그린 결과를 덜 걸러 내고, 참고 사진(📷) 설명도 쓸 수 없다. GPU가 허락하면 Large를 쓴다.

**VRAM 사용량 실측** (RTX 4060 Laptop 8GB, 모델 하나씩 단독, 최대 reserved 기준):

| 모델 | 적재 방식 | 이미지 | VRAM |
|---|---|---|---|
| SDXL 인페인팅 (Large) | fp16, CPU 오프로드 | 1280x720 | 5.7GB |
| SDXL 인페인팅 (Large) | fp16, CPU 오프로드 | 1920x1080 | 6.4GB |
| SD1.5 인페인팅 (Small) | fp16, CPU 오프로드 | 768x576 | 2.7GB |
| Qwen2.5-3B (LLM) | fp16 / 4bit | - | 5.8GB / 2.1GB |
| Qwen2.5-VL-3B (VLM) | fp16 / 4bit | 1280x720 | 7.3GB / 2.9GB |
| SAM2.1-small + Grounding DINO tiny | fp32 | 1920x1080 | 2.5GB |

VRAM 크기에 따른 적재 방식(fp16/4bit, 오프로드)은 앱이 자동으로 고른다. 기준은 [예상 시간과 PC별 자동 설정](#예상-시간과-pc별-자동-설정)을 본다.

### Background Swap: 처리 속도

RTX 4060 Laptop 실측, 인페인팅 생성 단계만, 30 step 기준. 변형 수(`num_variants`)와 재시도만큼 곱해진다.

| 이미지 | SDXL (Large) | SDXL + 빠른 모드 | SD1.5 (Small) |
|---|---|---|---|
| 768x576 | 약 13초/장 | | 약 8초/장 |
| 1024x768 | 약 19초/장 | 약 7초/장 | |
| 1280x720 | 약 23초/장 | | |
| 1920x1080 | 약 51초/장 | | |

- 인페인팅 최대 해상도 기본값은 긴 변 1280px다. 원본이 더 크면 줄여서 생성한 뒤 원본 해상도로 복원한다.
- 입력이 많으면 `frame_stride`(N장마다 1장 선택)로 줄인다. 탭 맨 아래에 그 PC 기준 예상 시간이 나온다.
- 노트북 GPU는 발열로 느려질 수 있다.

---

## Background Swap (LLM 기반 배경 교체 증강)

YOLO 라벨이 있는 이미지에서 **라벨 객체와 "유지할 배경"(경기장 필드/매트, 라인, 펜스, 골대 등)은 원본 픽셀 그대로 두고**, 나머지 배경(관중석, 벽, 천장, 조명 등)만 인페인팅 모델로 새로 그려서 다양한 대회장 배경의 학습 이미지를 만든다. 라벨은 원본 것을 그대로 복사하므로 객체·유지 영역 픽셀이 원본과 동일해야 하고, 이는 검증 단계에서 확인한다.

처리는 단계별 일괄 방식이다 (마스크 -> 프롬프트 생성 LLM -> 인페인팅 -> 검증). 한 단계에서 한 종류의 모델만 GPU에 올리므로 8GB VRAM에서 동작한다. 모든 설정은 GUI에서 만든다 (설정 파일을 직접 편집하지 않는다).

### 요구 사항

- 필요한 OS, Python, GPU, 디스크와 처리 속도는 위 [사양](#사양)을 본다.
- **첫 실행 시 모델을 자동 다운로드한다 (Large 합계 약 24GB).** 저장 위치는 `~/.cache/huggingface`이며 git 저장소에는 들어가지 않는다. 디스크 여유 공간을 확인할 것. 기본 모델은 로그인 없이 받을 수 있다. 로그인이 필요한 모델로 바꿨다면 `hf auth login`을 먼저 실행한다.

### 설치

GUI가 워커 프로세스를 **GUI와 같은 파이썬**으로 실행하므로, 설치한 venv의 파이썬으로 GUI를 띄워야 한다.

```bash
cd ~/ws/src/dataset_helper
make deps   # 처음 한 번 (tkinter 등 시스템 패키지)
make venv         # Large: requirements.txt + requirements-bg-large.txt
make venv-small   # Small: requirements.txt + requirements-bg-small.txt
```

| | Large (기본) | Small |
|---|---|---|
| 의존성 | `requirements-bg-large.txt` = Small + `bitsandbytes`(4bit), `peft`(빠른 모드) | `requirements-bg-small.txt` |
| 인페인팅 | SDXL (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) | SD1.5 (`stable-diffusion-v1-5/stable-diffusion-inpainting`, 512px) |
| LLM | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |
| 세그멘터 | `facebook/sam2.1-hiera-small` | `facebook/sam2.1-hiera-tiny` |
| VLM 검수 | 켬 (`Qwen/Qwen2.5-VL-3B-Instruct`) | 끔 |
| 빠른 모드 | 사용 가능 | 없음 (이미 빠름) |
| GPU | 8GB 기준 | 4GB급 |

- 모델 크기는 GUI의 "생성" 그룹에서 `Large` / `Small`로 고른다. Large 설치에서 Small 모델을 써도 된다. Small 설치에서는 4bit와 빠른 모드를 쓸 수 없다.
- `torch`는 requirements에 들어 있지 않다. 사용하는 머신의 CUDA 버전에 맞는 torch를 따로 설치한다 (실측 환경: torch 2.13 + cu130, `--system-site-packages`로 시스템의 torch를 그대로 사용).

### 실행

```bash
cd ~/colcon_ws/src/dataset_helper
make augmentor
```

venv를 거치지 않고 시스템 `python3 main.py`로 띄우면 워커에 의존성이 없어 설치 안내 메시지(`make venv` 또는 `make venv-small`)가 나오고 실행되지 않는다.

### 사용 순서

GUI에서 "Background Swap" 탭을 연다.

1. **폴더 지정** ("데이터" 그룹): `Images folder`, `Labels folder`, `Output folder`. 클래스 이름 파일(`Classes file`)을 지정하면 새 객체 검사와 클래스별 recall을 사용할 수 있다 (라벨 폴더 근처에 있으면 자동으로 채워진다).
2. **유지할 배경 항목 추가** ("유지할 배경" 그룹의 `추가` 버튼). 항목마다 방식(`color` / `text`)을 고른다.
   - **색상 영역** (`color`): 필드/매트처럼 넓은 단색 영역. `미리보기에서 클릭` 후 미리보기 이미지에서 해당 영역을 클릭하면 그 색을 기준으로 범위가 설정된다.
   - **텍스트 검출** (`text`): 펜스, 골대처럼 물체 형태. 검출용 프롬프트(단어/구, 예: `goal post` 또는 `골대`)를 입력한다.
   - `필수` 체크: 필수 항목을 못 찾은 이미지는 건너뛴다. 필수가 아니면 그 항목 없이 진행하고 경고만 기록한다.
3. 생성 파라미터(변형 수, 최대 시도 횟수 등)와 **배경 조합**을 확인한다.
   - 배경 조합은 새 배경을 고르는 메뉴판이다. 항목은 장소, 조명, 바닥, 주변 풍경, 사진 느낌 다섯 가지이고, 항목마다 후보 값을 `;`로 구분해 적는다. 새 배경을 하나 그릴 때마다 **체크한 항목**에서 후보 하나씩을 무작위로 골라 조합한다 (예: 대학 강당 + 따뜻한 조명 + 회색 카펫). 체크를 끈 항목은 쓰지 않지만 적어 둔 값은 남는다.
   - 칸에 흐린 회색으로 보이는 글자는 예시일 뿐 실제 입력이 아니다. 칸을 클릭하면 사라지고, 칸을 비워 두면 그 예시 값을 그대로 쓴다.
   - **참고 사진 (선택)**: 항목 줄의 `📷 사진`으로 그 항목의 참고 사진을 고를 수 있다 (여러 장 가능). VLM(Qwen2.5-VL)이 사진에서 그 항목에 해당하는 부분만 짧은 영어 문구로 설명하고(예: 조명 줄이면 조명만), 그 문구를 그 항목의 후보 값에 더한다. 칸이 비어 있으면 예시 대신 사진 설명만 쓴다. 설명은 Preview 뒤 창으로 보여 주고 `{output_dir}/references.json`에 저장해 Run이 그대로 쓴다. 고르지 않으면 예전과 같다.
   - **빠른 모드** ("생성" 그룹, Large 전용): 인페인팅에 Hyper-SD 8단계 LoRA(`ByteDance/Hyper-SD`의 `Hyper-SDXL-8steps-CFG-lora`)를 붙여 30단계 대신 8단계로 그린다. 약 2.7배 빠르고 배경 디테일이 조금 거칠어질 수 있다. 네거티브 프롬프트는 그대로 동작한다.
   - 유지 항목 이름, 검출 프롬프트, 배경 조합 후보 값, 클래스 이름은 한국어로 써도 된다. 아래 "한국어 입력" 참고.
4. **`Preview N=3`**: 무작위 3장에 전체 파이프라인을 돌려 `{output_dir}/_preview/`에 만들고 4패널(원본 / 마스크 / 생성 결과 / 차이 이미지)로 보여 준다. 마스크 색: 빨강=객체, 주황=객체 보호 링/그림자, 초록=유지 배경, 무색=새로 그릴 영역. **대량 실행 전에 마스크가 맞는지 반드시 확인한다.** 유지 항목이 틀리면 1~3으로 돌아가 고친다.
5. **`▶ Run`**: 설정 검증 후 `run_settings.json`을 기록하고 워커를 실행한다. **`■ Stop`**으로 중단할 수 있고, 같은 출력 폴더로 다시 Run하면 이어서 진행한다. 생성 설정을 바꾼 채 같은 출력 폴더에서 다시 Run하면 확인창이 뜨며 `새 출력 폴더` / `덮어쓰기` / `취소` 중에서 고른다.
6. **`Report`**: 상태별(accepted / rejected / skipped / exhausted) 개수, 거부 사유, 배경 조합 항목별 후보 사용 분포, 클래스별 GDINO recall을 본다.
7. **`Export`**: Accepted 결과를 지정한 학습 데이터셋의 images / labels 폴더로 복사한다. 같은 이름의 파일이 있으면 덮어쓰지 않고 건너뛰며 건너뛴 개수를 알려 준다.

### 예상 시간과 PC별 자동 설정

- 탭 맨 아래에 **Run 예상 시간과 완료 시각**이 나온다. 이미지 수, `Frame stride`, `Variants`, 해상도, LLM/VLM 사용 여부를 바꾸면 바로 다시 계산한다. 첫 시도 기준이며, 검수에서 떨어져 다시 그리면 최대 `Max attempts` 배까지 늘 수 있다.
- 실행 중에는 실제 진행 속도로 **남은 시간과 완료 예상 시각**을 계속 고쳐 쓴다.
- 앱을 켤 때 GPU 이름, VRAM, 계산 속도를 1~3초 동안 재서 예상 시간을 그 PC에 맞춘다. Preview나 Run을 한 번 돌리면 그 PC에서 실제로 걸린 속도로 바꿔 쓴다 (`~/.config/dataset_helper/hardware_profile.json`).
- 모델 적재 방식도 VRAM 크기로 고른다: LLM은 7.5GB 이상이면 fp16, VLM은 11GB 이상이면 fp16(아니면 4bit), 인페인팅은 15GB 이상이면 전부 GPU, 7GB 이상이면 일부 CPU 오프로드, 그보다 작으면 순차 오프로드(느림).
- 오른쪽 아래 **`ⓘ Info`** 버튼: 모든 항목 설명, 빨리 끝내는 방법(빠른 모드, `Frame stride`, `Variants`, `Max attempts`, 해상도, VLM), 이 PC에서 정해진 값을 보여 준다. 값은 자동으로 바뀌지 않는다.
- 터미널에서 실행했다면 `Ctrl+C`로 창을 닫을 수 있다.

### 한국어 입력

인페인팅 모델(SDXL)과 검출기(Grounding DINO)는 영어만 알아듣는다. 그래서 Preview나 Run을 시작할 때 한글이 든 입력(유지 항목 이름과 검출 프롬프트, 배경 조합 후보 값, 클래스 이름 파일의 이름)을 LLM(`Qwen/Qwen2.5-3B-Instruct`)으로 한 번 영어로 바꾸고, 이후 모든 단계는 영어를 쓴다.

- **Preview가 끝나면 "번역 결과" 창에 `원문 → 번역`을 보여 준다.** 번역이 틀렸으면 그 항목을 영어로 직접 입력하고 다시 Preview한다.
- 번역은 `{output_dir}/translations.json`에 저장되어 Run과 이어하기가 같은 번역을 쓴다. 저장된 번역이 있으면 LLM을 다시 불러오지 않는다.
- `Use LLM`을 꺼도 한글 입력이 있으면 번역에는 LLM을 쓴다 (모델을 받아 둬야 한다).
- 번역 결과가 비었거나 영문이 아니면 실행을 멈추고 어떤 항목인지 알려 준다.
- 리포트의 클래스별 recall, 경고 문구, 배경 조합 분포의 후보 값은 번역된 영어 이름으로 나온다.

### 출력 폴더 구조

```
{output_dir}/
├── run_settings.json      # GUI가 자동 기록하는 실행 설정 (직접 편집하지 않음)
├── translations.json      # 한국어 입력의 영어 번역 (한글 입력이 있을 때만)
├── references.json        # 참고 사진 설명 (참고 사진을 골랐을 때만)
├── items.jsonl            # 선택된 원본 목록
├── masks/                 # 이미지별 마스크 (object_core / object_protect / keep / inpaint)
├── jobs.jsonl             # 변형별 시드, 장면, 프롬프트
├── generated/             # 합성 완료된 생성 이미지
├── Accepted/
│   ├── images/{stem}_bgswap{n}.jpg
│   ├── labels/{stem}_bgswap{n}.txt     # 원본 라벨 복사
│   └── previews/{stem}_bgswap{n}.jpg   # bbox 표시본 (검토용)
├── Rejected/              # 검증 탈락 이미지
├── Manifest.jsonl         # 시도마다 1줄: 시드, 장면, 프롬프트, 상태, 거부 사유, 검사 결과
└── report.json
```

### 주의 사항

- **train 이미지만 사용한다.** val/test 이미지에서 만든 증강본이 학습 데이터에 섞이면 평가가 무의미해진다. 입력 또는 Export 경로에 `val`, `valid`, `validation`, `test` 폴더 이름이 들어 있으면 GUI가 경고 확인창을 띄운다. 의도한 경우가 아니면 취소한다.
- **새 객체 검사는 위험을 줄이는 필터이지 보증이 아니다.** 생성된 배경에 라벨 대상 객체가 새로 생기면 라벨 누락이 되어 오히려 학습을 해친다. 이를 Grounding DINO와 VLM으로 걸러내지만, 검출기가 못 찾는다고 해서 새 객체가 없다는 뜻은 아니다. 특히 작은 객체 클래스는 recall이 낮을 수 있으니 Report의 클래스별 recall을 확인하고, Accepted/previews를 일부 직접 눈으로 검토한 뒤 Export한다.
- 유지 항목의 색상 방식은 배경에 비슷한 색이 있으면 과검출될 수 있고, 텍스트 방식은 검출기 품질에 좌우된다. Preview로 확인한다.
- VRAM 부족(OOM)이 나면 "고급 설정"(기본 접힘)의 `Max resolution px`를 낮추거나 다른 GPU 사용 프로그램을 종료한다.
- 원본 객체와 새 배경의 조명이 어긋날 수 있다 (학습 다양성 목적상 허용).

### 모델 라이선스

기본으로 쓰는 모델의 라이선스를 사용 전에 각 모델 카드에서 확인한다.

| 용도 | 기본 모델 | 라이선스 |
|---|---|---|
| 장면 프롬프트 생성 (LLM) | `Qwen/Qwen2.5-3B-Instruct` | 비상업 연구용으로 알려져 있음, 모델 카드 확인 필요 |
| 결과 검증 (VLM) | `Qwen/Qwen2.5-VL-3B-Instruct` | 위와 동일 |
| 인페인팅 | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | CreativeML OpenRAIL++-M |

상업적 용도라면 허용 라이선스의 모델로 모델 ID를 교체한다.

### 효과 평가 방법 (수동 절차)

배경 다양성이 새 환경에서의 일반화를 실제로 높이는지는 아래 방식으로 확인한다. 이 도구가 자동으로 해 주지는 않는다.

1. 학습에 쓰지 않는 **다른 환경(다른 대회장/공간/조명)에서 촬영한 소량의 테스트셋**을 따로 확보해 라벨링한다.
2. 같은 학습 설정으로 (a) 원본만, (b) 원본 + Background Swap 증강본으로 각각 학습한다.
3. 기존 val셋과 새 환경 테스트셋 양쪽에서 mAP를 비교한다. 목표는 새 환경 테스트셋에서 (b)가 개선되고, 기존 val셋 성능이 크게 떨어지지 않는 것이다.
4. 합성 비율(원본:증강 = 1:0.5, 1:1, 1:2)을 바꿔 가며 비교한다.
