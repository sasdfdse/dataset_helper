#!/bin/bash
# VideoDatasetTool 빌드 스크립트
# Ubuntu / Debian 계열 기준

set -e

echo "=== VideoDatasetTool 빌드 ==="

# FFmpeg 개발 라이브러리 설치 확인
if ! pkg-config --exists libavcodec 2>/dev/null; then
    echo "[*] FFmpeg 개발 라이브러리 설치 중..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        libavcodec-dev libavformat-dev libavutil-dev \
        libswscale-dev libswresample-dev
fi

# 컴파일
echo "[*] 컴파일 중..."
g++ -std=c++17 -O2 dataset_helper/dataset_helper.cpp -o video_tool \
    $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale libswresample) \
    -lpthread

echo "[✓] 빌드 완료: ./video_tool"
echo ""

# Qt GUI 빌드
echo "[*] Qt GUI 빌드 중..."
cd dataset_helper
qmake video_tool_qt.pro -o Makefile_qt 2>/dev/null && make -f Makefile_qt -s
if [ $? -eq 0 ]; then
    mv video_tool_qt ../ 2>/dev/null || true
    echo "[✓] Qt GUI 빌드 완료: ./video_tool_qt"
else
    echo "[!] Qt GUI 빌드 실패 (Qt5 개발 패키지 필요: sudo apt install qtbase5-dev)"
fi
cd ..

echo ""
echo "실행 (터미널): ./video_tool"
echo "실행 (GUI):    ./video_tool_qt"
