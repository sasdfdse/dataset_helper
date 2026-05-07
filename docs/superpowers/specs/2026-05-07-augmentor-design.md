# Dataset Helper GUI — Design Spec
Date: 2026-05-07

## Overview

기존 C++ CLI 기반 dataset_helper(VideoDatasetTool)의 기능과 새로운 Augmentation 기능을
Python tkinter GUI 단일 앱으로 통합한 툴.
탭 구조로 두 기능을 분리하여 제공.

## Package Structure

```
augmentor/
├── main.py
├── requirements.txt
├── gui/
│   ├── __init__.py
│   ├── app.py               ← 메인 윈도우, 탭(Notebook) 구성
│   ├── video_tab.py         ← Video Tool 탭 UI
│   └── augment_tab.py       ← Augmentation 탭 UI (SettingsPanel + PreviewPanel)
└── core/
    ├── __init__.py
    ├── augment.py           ← brightness, blur, noise, scale, cutmix 함수
    ├── label.py             ← YOLO .txt 읽기/쓰기/bbox 변환
    ├── pipeline.py          ← augmentation 배치 처리 오케스트레이터
    └── video.py             ← ffmpeg-python 기반 변환/추출 로직
```

## GUI Layout

최상단 탭(ttk.Notebook)으로 두 기능 분리:

```
┌────────────────────────────────────────────┐
│  [📹 Video Tool]  [🔧 Augmentation]         │
├────────────────────────────────────────────┤
│  ... 각 탭 내용 ...                          │
└────────────────────────────────────────────┘
```

### Video Tool 탭

```
┌─────────────────────────────────────────────┐
│  입력 파일: [경로...............]  [Browse]   │
│  출력 경로: [경로...............]  [Browse]   │
│                                             │
│  ○ WEBM → MP4 변환                          │
│  ○ 동영상 → 프레임 추출                       │
│  ○ WEBM → MP4 → 프레임 추출 (연속)           │
│  ○ 동영상 정보 보기                           │
│                                             │
│  [프레임 추출 옵션] (추출 모드일 때만 활성화)  │
│    추출 비율: ○ 전체  ○ 1/2  ○ 1/3          │
│    그레이스케일: □                           │
│                                             │
│  [동영상 정보] (정보 보기 선택 시 여기 표시)  │
│  ─────────────────────────────────────────  │
│  [████████████░░░░] 42%                     │
│              [▶ Run]                        │
└─────────────────────────────────────────────┘
```

### Augmentation 탭

```
┌─────────────────┬───────────────────────────┐
│  [설정 패널]     │  [프리뷰 패널]              │
│                 │                            │
│  📁 Dataset     │  Original    Augmented     │
│  images/ [...]  │  ┌────────┐  ┌────────┐   │
│  labels/  [...]  │  │ +bbox  │  │ +bbox  │   │
│                 │  └────────┘  └────────┘   │
│  ─────────────  │                            │
│  ✅ Brightness  │  < prev         next >     │
│     min [-100]  │                            │
│     max [+100]  │  ────────────────────────  │
│                 │  [████████░░░░] 42/100     │
│  ✅ Blur        │                            │
│     radius [3]  ├───────────────────────────┤
│                 │                            │
│  ✅ Noise       │    [▶ Run Augmentation]    │
│     strength[25]│                            │
│                 └───────────────────────────┘
│  ✅ Scale
│     min [0.8]
│     max [1.2]
│
│  ✅ CutMix
│     pairs [10]
│
│  배수: [x3] 장
└─────────────────
```

## Video Tool 기능 스펙

기존 C++ dataset_helper.cpp 기능을 그대로 Python으로 포팅. `ffmpeg-python` 라이브러리 사용.

| 기능 | 설명 |
|---|---|
| WEBM → MP4 변환 | ffmpeg-python으로 재인코딩, 진행률 표시 |
| 프레임 추출 | 비율(1/1, 1/2, 1/3), 그레이스케일 옵션, JPEG 저장 |
| 연속 작업 | WEBM→MP4 후 바로 프레임 추출 |
| 동영상 정보 | 포맷, 길이, fps, 해상도, 코덱, 비트레이트 표시 |

- 진행률: ffmpeg stderr 파싱하여 Progressbar 업데이트
- 출력 기본값: 입력 파일과 같은 디렉토리, stem + 확장자

## Augmentation 기능 스펙

### Input / Output (Roboflow 구조)

```
dataset/train/
  images/ball_001.jpg          ← 원본 입력
  labels/ball_001.txt          ← 원본 라벨
  images/ball_001_aug1.jpg     ← augmented 출력
  labels/ball_001_aug1.txt     ← 변환된 라벨 출력
```

- 같은 폴더에 `_aug{n}` suffix로 저장
- 배수 N: 원본 1장당 N장 생성 (체크된 augmentation 중 랜덤 조합)

### Augmentation 종류

**Brightness**
- 파라미터: min, max (−100 ~ +100, 픽셀 오프셋)
- 바운딩박스: 변환 없음

**Blur**
- 파라미터: radius (1 ~ 15, Gaussian kernel size, 홀수 강제)
- 바운딩박스: 변환 없음

**Noise**
- 파라미터: strength (0 ~ 100, Gaussian noise std)
- 바운딩박스: 변환 없음

**Scale**
- 파라미터: min, max (0.5 ~ 2.0 배율)
- 확대(>1.0): 이미지 키운 뒤 중앙 crop → bbox 좌표 역산
- 축소(<1.0): 이미지 줄인 뒤 검정 패딩 → bbox 좌표 오프셋 추가
- bbox가 crop 영역 밖으로 완전히 나간 경우 해당 라벨 삭제
- 바운딩박스: x_center, y_center, width, height 재계산 (YOLO normalized)

**CutMix**
- 파라미터: pairs (생성할 쌍 수, 기본 10)
- 동작: 같은 폴더에서 랜덤 두 이미지 선택 → 랜덤 직사각형 영역을 이미지 B로 덮어씌움
- A bbox: 덮인 영역과 교차하면 clip, 원래 면적의 30% 미만으로 잘리면 삭제
- B bbox: 붙여넣은 영역 안으로 좌표 변환 후 추가
- 출력 파일명: `{imgA_stem}_{imgB_stem}_cutmix_{n}.jpg`

## Core Module Responsibilities

| 모듈 | 역할 |
|---|---|
| `core/video.py` | ffmpeg-python 래핑, 변환/추출/정보 함수, 진행률 콜백 |
| `core/label.py` | YOLO .txt 읽기/쓰기, bbox clip/변환 함수 |
| `core/augment.py` | 각 augmentation 함수 (numpy array 입출력) |
| `core/pipeline.py` | 폴더 순회, 파일명 생성, augment+label 변환 후 저장, 진행률 콜백 |
| `gui/app.py` | 메인 윈도우, ttk.Notebook 탭 조합 |
| `gui/video_tab.py` | Video Tool 탭 UI 및 이벤트 핸들러 |
| `gui/augment_tab.py` | Augmentation 탭 UI (SettingsPanel + PreviewPanel) |

## Dependencies

```
opencv-python
numpy
Pillow
ffmpeg-python
```

(tkinter는 python3-tk 패키지로 이미 설치됨)
(ffmpeg 바이너리는 시스템에 설치돼 있어야 함)

## Key Behaviors

- Run 버튼 클릭 시 작업을 별도 스레드에서 실행 (GUI 블로킹 방지)
- 진행률은 콜백으로 GUI에 전달 (`root.after()` 사용)
- augmentation 완료 후 프리뷰에서 결과 이미지 탐색 가능
- 체크되지 않은 augmentation은 건너뜀
- Video Tool에서 모드 선택에 따라 옵션 UI 동적 활성화/비활성화
