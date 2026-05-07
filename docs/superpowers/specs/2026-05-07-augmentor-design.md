# Augmentor GUI — Design Spec
Date: 2026-05-07

## Overview

Roboflow에서 라벨링한 YOLO 데이터셋에 augmentation을 적용하는 Python tkinter GUI 툴.
라벨 파일(.txt)도 함께 변환하여 Roboflow export 구조와 동일한 형태로 저장.

## Package Structure

```
augmentor/
├── main.py
├── requirements.txt
├── gui/
│   ├── __init__.py
│   ├── app.py
│   └── panels.py
└── core/
    ├── __init__.py
    ├── augment.py
    ├── label.py
    └── pipeline.py
```

## GUI Layout

3패널 레이아웃 (tkinter):

- **왼쪽 — SettingsPanel**: 폴더 선택(images/, labels/), augmentation 체크박스 + 파라미터 슬라이더, 배수 입력
- **오른쪽 상단 — PreviewPanel**: 원본/augmented 나란히 표시, bbox 오버레이, prev/next 탐색
- **오른쪽 하단**: 진행률 바(ttk.Progressbar) + Run 버튼

## Input / Output

Roboflow export 구조를 그대로 사용:

```
dataset/train/
  images/ball_001.jpg          ← 입력
  labels/ball_001.txt          ← 입력
  images/ball_001_aug1.jpg     ← 출력
  labels/ball_001_aug1.txt     ← 출력
```

- 같은 폴더에 `_aug{n}` suffix로 저장
- 배수 설정: 원본 1장당 N장 생성 (각 augmentation 조합 랜덤 선택)

## Augmentation Specs

### Brightness
- 파라미터: min, max (범위: -100 ~ +100, 픽셀 오프셋)
- 바운딩박스: 변환 없음

### Blur
- 파라미터: radius (1 ~ 15, Gaussian kernel size, 홀수)
- 바운딩박스: 변환 없음

### Noise
- 파라미터: strength (0 ~ 100, Gaussian noise std)
- 바운딩박스: 변환 없음

### Scale
- 파라미터: min, max (0.5 ~ 2.0, 배율)
- 확대(>1.0): 이미지 키운 뒤 중앙 crop → bbox 좌표 역산
- 축소(<1.0): 이미지 줄인 뒤 검정 패딩 → bbox 좌표 오프셋 추가
- bbox가 crop 영역 밖으로 완전히 나간 경우 해당 라벨 삭제
- 바운딩박스: x_center, y_center, width, height 재계산 (YOLO normalized)

### CutMix
- 파라미터: pairs (생성할 쌍 수, 기본 10)
- 동작: 같은 폴더에서 랜덤 두 이미지 선택 → 랜덤 직사각형 영역을 이미지 B로 덮어씌움
- A bbox: 덮인 영역과 교차하는 bbox는 clip, 원래 면적의 30% 미만으로 잘리면 삭제
- B bbox: 붙여넣은 영역 안으로 좌표 변환 후 추가
- 출력 파일명: `{imgA}_{imgB}_cutmix_{n}.jpg`

## Core Module Responsibilities

| 모듈 | 역할 |
|---|---|
| `core/label.py` | YOLO .txt 읽기/쓰기, bbox clip/변환 함수 |
| `core/augment.py` | 각 augmentation 함수 (이미지 numpy array 입출력) |
| `core/pipeline.py` | 폴더 순회, 파일명 생성, augment+label 변환 후 저장, 진행률 콜백 |
| `gui/panels.py` | SettingsPanel, PreviewPanel 위젯 |
| `gui/app.py` | 메인 윈도우, 패널 조합, Run 버튼 핸들러 |

## Dependencies

```
opencv-python
numpy
Pillow
```

(tkinter는 python3-tk 패키지로 이미 설치됨)

## Key Behaviors

- Run 버튼 클릭 시 pipeline을 별도 스레드에서 실행 (GUI 블로킹 방지)
- 진행률은 콜백으로 GUI에 전달 (`after()` 사용)
- 완료 후 프리뷰에서 생성된 augmented 이미지 탐색 가능
- 체크되지 않은 augmentation은 건너뜀
- augmentation은 랜덤 조합으로 적용 (배수만큼 반복)
