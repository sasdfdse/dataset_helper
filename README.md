# dataset_helper


**This is a helpful package for changing video encoding and creating datasets.**

This package has two functions: one converts .webm files to .mp4, and the other extracts video frames and saves them as RGB or grayscale images.

**python dependency install :** 

1. pip install -r requirements.txt

**build :**
1. cd ~/ws/src/dataset_helper
2. bash build.sh

**run :** 
1. cd ~/ws/src/dataset_helper/augmentor
2. python3 main.py

(Background Swap 기능을 쓸 때는 아래 "Background Swap" 섹션의 venv 설치/실행 방법을 따른다.)

---

## Background Swap (LLM 기반 배경 교체 증강)

YOLO 라벨이 있는 이미지에서 **라벨 객체와 "유지할 배경"(경기장 필드/매트, 라인, 펜스, 골대 등)은 원본 픽셀 그대로 두고**, 나머지 배경(관중석, 벽, 천장, 조명 등)만 인페인팅 모델로 새로 그려서 다양한 대회장 배경의 학습 이미지를 만든다. 라벨은 원본 것을 그대로 복사하므로 객체·유지 영역 픽셀이 원본과 동일해야 하고, 이는 검증 단계에서 확인한다.

처리는 단계별 일괄 방식이다 (마스크 -> 프롬프트 생성 LLM -> 인페인팅 -> 검증). 한 단계에서 한 종류의 모델만 GPU에 올리므로 8GB VRAM에서 동작한다. 모든 설정은 GUI에서 만든다 (설정 파일을 직접 편집하지 않는다).

### 요구 사항

- 실측 환경: RTX 4060 Laptop (VRAM 8GB), RAM 62GB. 이보다 작은 GPU는 검증하지 않았다.
- 인페인팅 최대 해상도 기본값은 긴 변 1280px. 원본이 더 크면 줄여서 생성한 뒤 원본 해상도로 복원한다.
- 처리 시간 (SDXL 인페인팅, 30 step, 생성 단계만): 1280x720 약 23초/장, 1920x1080 약 51초/장. 변형 수(`num_variants`)와 재시도만큼 곱해진다. 대량 실행은 장시간 실행을 전제로 하고, 입력이 많으면 `frame_stride`(N장마다 1장 선택)로 줄인다.
- **첫 실행 시 모델을 자동 다운로드한다 (합계 약 22GB).** 저장 위치는 `~/.cache/huggingface`이며 git 저장소에는 들어가지 않는다. 디스크 여유 공간을 확인할 것. 기본 모델은 로그인 없이 받을 수 있다. 로그인이 필요한 모델로 바꿨다면 `hf auth login`을 먼저 실행한다.

### 설치

GUI가 워커 프로세스를 **GUI와 같은 파이썬**으로 실행하므로, 설치한 venv의 파이썬으로 GUI를 띄워야 한다.

```bash
cd ~/ws/src/dataset_helper/augmentor
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt -r requirements-bg.txt
```

- `torch`는 `requirements-bg.txt`에 들어 있지 않다. 사용하는 머신의 CUDA 버전에 맞는 torch를 따로 설치한다 (실측 환경: torch 2.13 + cu130, `--system-site-packages`로 시스템의 torch를 그대로 사용).
- `tkinter`가 필요하다 (`sudo apt install python3-tk`).

### 실행

```bash
cd ~/ws/src/dataset_helper/augmentor
.venv/bin/python main.py
```

시스템 `python3 main.py`로 띄우면 워커에 의존성이 없어 설치 안내 메시지(`pip install -r requirements-bg.txt`)가 나오고 실행되지 않는다.

### 사용 순서

GUI에서 "Background Swap" 탭을 연다.

1. **폴더 지정** ("데이터" 그룹): Images 폴더, Labels 폴더, 출력 폴더. 클래스 이름 파일(`Classes file`)을 지정하면 새 객체 검사와 클래스별 recall을 사용할 수 있다 (라벨 폴더 근처에 있으면 자동으로 채워진다).
2. **유지할 배경 항목 추가** ("유지할 배경" 그룹). 항목마다 방식을 고른다.
   - **색상 영역**: 필드/매트처럼 넓은 단색 영역. 미리보기 이미지에서 해당 영역을 클릭하면 그 색을 기준으로 범위가 설정된다.
   - **텍스트 검출**: 펜스, 골대처럼 물체 형태. 검출용 프롬프트(영어 단어/구, 예: `goal post`)를 입력한다.
   - **필수** 체크: 필수 항목을 못 찾은 이미지는 건너뛴다. 필수가 아니면 그 항목 없이 진행하고 경고만 기록한다.
3. 생성 파라미터(변형 수, 최대 시도 횟수 등)와 장면 축(대회장 종류, 조명 등. 값은 `;`로 구분)을 확인한다.
4. **Preview**: 무작위 3장에 전체 파이프라인을 돌려 `{output_dir}/_preview/`에 만들고 4패널(원본 / 마스크 / 생성 결과 / 차이 이미지)로 보여 준다. 마스크 색: 빨강=객체, 주황=객체 보호 링/그림자, 초록=유지 배경, 무색=새로 그릴 영역. **대량 실행 전에 마스크가 맞는지 반드시 확인한다.** 유지 항목이 틀리면 1~3으로 돌아가 고친다.
5. **Run**: 설정 검증 후 `run_settings.json`을 기록하고 워커를 실행한다. **Stop**으로 중단할 수 있고, 같은 출력 폴더로 다시 Run하면 이어서 진행한다.
6. **Report**: 상태별(accepted / rejected / skipped / exhausted) 개수, 거부 사유, 장면 축 사용 분포, 클래스별 GDINO recall을 본다.
7. **Export**: Accepted 결과를 지정한 학습 데이터셋의 images / labels 폴더로 복사한다. 같은 이름의 파일이 있으면 덮어쓰지 않고 건너뛰며 건너뛴 개수를 알려 준다.

### 출력 폴더 구조

```
{output_dir}/
├── run_settings.json      # GUI가 자동 기록하는 실행 설정 (직접 편집하지 않음)
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

### CLI (참고)

GUI가 내부적으로 호출하는 워커이며 보통 직접 쓸 필요는 없다.

```bash
cd ~/ws/src/dataset_helper/augmentor
.venv/bin/python -m background_swap run     --settings {output_dir}/run_settings.json
.venv/bin/python -m background_swap preview --settings {output_dir}/run_settings.json --n 3
.venv/bin/python -m background_swap export  --settings {output_dir}/run_settings.json --images-dir IMAGES --labels-dir LABELS
```
