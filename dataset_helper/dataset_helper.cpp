/**
 * VideoDatasetTool
 * ─────────────────────────────────────────────────────────────────────────────
 *  • WEBM → MP4 변환 (FFmpeg 재인코딩)
 *  • 동영상 → 프레임 이미지 추출 (JPEG/PNG)
 *  • 추출 비율 선택: 전체(1/1), 절반(1/2), 3분의 1(1/3)
 *  • 그레이스케일 변환 옵션
 *
 * 빌드:
 *   g++ -std=c++17 main.cpp -o video_tool \
 *       $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale libswresample) \
 *       -lpthread
 *
 * 사용법: ./video_tool [옵션]
 *   옵션 없이 실행하면 대화형 메뉴가 뜸
 */

#include "video_core.hpp"
#include <iostream>
#include <iomanip>

// ─────────────────────── 색상 출력 헬퍼 ────────────────────────
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define MAGENTA "\033[35m"

void printBanner() {
    std::cout << CYAN << BOLD;
    std::cout << R"(
╔══════════════════════════════════════════════════════╗
║          VideoDatasetTool v1.0              ROBIT    ║
║  WEBM→MP4 변환 │ 프레임 추출 │ 그레이스케일 변환            ║
╚══════════════════════════════════════════════════════╝
)" << RESET << std::endl;
}

// ─────────────────────── 진행률 표시 ────────────────────────────
void printProgress(int current, int total, const std::string& label = "") {
    int width = 40;
    double ratio = (total > 0) ? (double)current / total : 0.0;
    int filled = (int)(ratio * width);

    std::cout << "\r" << GREEN << "[";
    for (int i = 0; i < width; i++)
        std::cout << (i < filled ? "█" : "░");
    std::cout << "] " << RESET
              << std::setw(5) << std::fixed << std::setprecision(1)
              << (ratio * 100.0) << "% "
              << "(" << current << "/" << total << ") "
              << label;
    std::cout.flush();
}


// ─────────────────────── 입력 헬퍼 ──────────────────────────────
std::string prompt(const std::string& msg) {
    std::cout << YELLOW << msg << RESET;
    std::string s;
    std::getline(std::cin, s);
    // 앞뒤 공백 제거
    auto b = s.find_first_not_of(" \t\"'");
    auto e = s.find_last_not_of(" \t\"'");
    return (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
}

int promptInt(const std::string& msg, int lo, int hi) {
    while (true) {
        std::string s = prompt(msg);
        try {
            int v = std::stoi(s);
            if (v >= lo && v <= hi) return v;
        } catch (...) {}
        std::cout << RED << "  ✗ " << lo << "~" << hi << " 범위의 숫자를 입력하세요.\n" << RESET;
    }
}

bool promptBool(const std::string& msg) {
    std::string s = prompt(msg + " (y/n): ");
    return (!s.empty() && (s[0] == 'y' || s[0] == 'Y'));
}

// ─────────────────────── 메인 메뉴 ──────────────────────────────
void menuConvert() {
    std::cout << "\n" << BOLD << "═══ WEBM → MP4 변환 ═══\n" << RESET;
    std::string inp = prompt("  입력 WEBM 파일 경로: ");
    if (inp.empty()) { std::cout << RED << "취소됨\n" << RESET; return; }

    // 기본 출력 경로: 입력과 같은 디렉토리, 확장자만 mp4
    fs::path p(inp);
    std::string defOut = (p.parent_path() / p.stem()).string() + ".mp4";
    std::cout << "  기본 출력 경로: " << CYAN << defOut << RESET << "\n";
    std::string out = prompt("  출력 MP4 경로 (Enter=기본값): ");
    if (out.empty()) out = defOut;

    std::cout << "\n" << GREEN << "  변환 시작...\n" << RESET;
    auto r = convertWebmToMp4(inp, out, [](int pct, const std::string& msg) {
        printProgress(pct, 100, msg);
    });
    std::cout << std::endl;
    if (r.success) {
        std::cout << GREEN << BOLD << "\n  ✓ 변환 성공!\n" << RESET;
        std::cout << "    출력 파일: " << out << "\n";
        std::cout << "    파일 크기: " << humanSize(r.outputSize) << "\n";
        std::cout << "    소요 시간: " << std::fixed << std::setprecision(1)
                  << r.elapsedSec << "초\n";
    } else {
        std::cout << RED << "\n  ✗ 오류: " << r.message << "\n" << RESET;
    }
}

void menuExtract() {
    std::cout << "\n" << BOLD << "═══ 프레임 추출 ═══\n" << RESET;
    std::string inp = prompt("  입력 동영상 파일 경로: ");
    if (inp.empty()) { std::cout << RED << "취소됨\n" << RESET; return; }

    std::string defDir = fs::path(inp).stem().string() + "_frames";
    std::cout << "  기본 출력 디렉토리: " << CYAN << defDir << RESET << "\n";
    std::string outDir = prompt("  출력 디렉토리 (Enter=기본값): ");
    if (outDir.empty()) outDir = defDir;

    std::cout << "\n  저장 비율 선택:\n";
    std::cout << "    1) 전체 (모든 프레임)\n";
    std::cout << "    2) 1/2 (프레임마다 1개씩 건너뜀)\n";
    std::cout << "    3) 1/3 (프레임마다 2개씩 건너뜀)\n";
    int ratio = promptInt("  선택 (1-3): ", 1, 3);

    bool gray = promptBool("  그레이스케일로 저장할까요?");

    ExtractOptions opts;
    opts.ratio     = ratio;
    opts.grayscale = gray;
    opts.format    = "jpg";
    opts.outDir    = outDir;

    std::cout << "\n" << GREEN << "  추출 시작...\n" << RESET;
    auto r = extractFrames(inp, opts, [](int pct, const std::string& msg) {
        printProgress(pct, 100, msg);
    });
    std::cout << std::endl;

    if (r.success) {
        std::cout << GREEN << BOLD << "\n  ✓ 추출 완료!\n" << RESET;
        std::cout << "    총 프레임 수: " << r.totalFrames << "\n";
        std::cout << "    저장된 이미지: " << r.savedFrames << "장\n";
        std::cout << "    출력 디렉토리: " << outDir << "\n";
        std::cout << "    소요 시간: " << std::fixed << std::setprecision(1)
                  << r.elapsedSec << "초\n";
    } else {
        std::cout << RED << "\n  ✗ 오류: " << r.message << "\n" << RESET;
    }
}

void menuConvertAndExtract() {
    std::cout << "\n" << BOLD << "═══ WEBM → MP4 변환 + 프레임 추출 (연속 작업) ═══\n" << RESET;
    std::string inp = prompt("  입력 WEBM 파일 경로: ");
    if (inp.empty()) { std::cout << RED << "취소됨\n" << RESET; return; }

    fs::path p(inp);
    std::string mp4Path = (p.parent_path() / p.stem()).string() + ".mp4";
    std::string outDir  = p.stem().string() + "_frames";

    std::cout << "  MP4 출력: " << CYAN << mp4Path << RESET << "\n";
    std::cout << "  프레임 디렉토리: " << CYAN << outDir << RESET << "\n";

    std::cout << "\n  저장 비율:\n"
              << "    1) 전체  2) 1/2  3) 1/3\n";
    int ratio = promptInt("  선택 (1-3): ", 1, 3);
    bool gray = promptBool("  그레이스케일?");

    // 1단계: 변환
    std::cout << "\n" << GREEN << "  [1/2] WEBM → MP4 변환 중...\n" << RESET;
    auto cr = convertWebmToMp4(inp, mp4Path, [](int pct, const std::string& msg) {
        printProgress(pct, 100, msg);
    });
    std::cout << std::endl;
    if (!cr.success) {
        std::cout << RED << "  ✗ 변환 실패: " << cr.message << "\n" << RESET;
        return;
    }
    std::cout << GREEN << "  ✓ MP4 변환 완료 (" << humanSize(cr.outputSize) << ")\n" << RESET;

    // 2단계: 추출
    std::cout << "\n" << GREEN << "  [2/2] 프레임 추출 중...\n" << RESET;
    ExtractOptions opts{ratio, gray, "jpg", outDir};
    auto er = extractFrames(mp4Path, opts, [](int pct, const std::string& msg) {
        printProgress(pct, 100, msg);
    });
    std::cout << std::endl;
    if (er.success) {
        std::cout << GREEN << BOLD << "\n  ✓ 모든 작업 완료!\n" << RESET;
        std::cout << "    저장된 이미지: " << er.savedFrames << "장\n";
        std::cout << "    출력 디렉토리: " << outDir << "\n";
    } else {
        std::cout << RED << "  ✗ 추출 실패: " << er.message << "\n" << RESET;
    }
}

// ─────────────────────── 동영상 정보 보기 ───────────────────────
void menuInfo() {
    std::cout << "\n" << BOLD << "═══ 동영상 정보 ═══\n" << RESET;
    std::string inp = prompt("  파일 경로: ");
    if (inp.empty()) return;

    VideoInfo info = getVideoInfo(inp);
    if (!info.valid) {
        std::cout << RED << "  오류: " << info.error << "\n" << RESET; return;
    }
    std::cout << "\n  파일: " << inp << "\n";
    std::cout << "  포맷: " << info.format << "\n";
    std::cout << "  길이: " << info.durationSec/60 << "분 " << info.durationSec%60 << "초\n";
    if (info.bitrate > 0)
        std::cout << "  비트레이트: " << info.bitrate/1000 << " kbps\n";
    if (!info.videoCodec.empty()) {
        std::cout << "  [비디오] " << info.videoCodec
                  << " | " << info.width << "×" << info.height;
        if (info.fps > 0)
            std::cout << " | " << std::fixed << std::setprecision(2) << info.fps << " fps";
        if (info.nbFrames > 0)
            std::cout << " | 총 프레임: " << info.nbFrames;
        std::cout << "\n";
    }
    if (!info.audioCodec.empty()) {
        std::cout << "  [오디오] " << info.audioCodec
                  << " | " << info.sampleRate << " Hz"
                  << " | " << info.channels << "ch\n";
    }
}

// ─────────────────────── main ───────────────────────────────────
int main(int argc, char* argv[]) {
    // FFmpeg 로그 최소화
    av_log_set_level(AV_LOG_ERROR);

    printBanner();

    while (true) {
        std::cout << "\n" << BOLD << "메뉴\n" << RESET;
        std::cout << "  1) WEBM → MP4 변환\n";
        std::cout << "  2) 동영상 → 프레임 이미지 추출\n";
        std::cout << "  3) WEBM → MP4 변환 후 바로 프레임 추출\n";
        std::cout << "  4) 동영상 정보 보기\n";
        std::cout << "  0) 종료\n";

        int choice = promptInt("\n선택: ", 0, 4);

        switch (choice) {
            case 1: menuConvert(); break;
            case 2: menuExtract(); break;
            case 3: menuConvertAndExtract(); break;
            case 4: menuInfo(); break;
            case 0:
                std::cout << CYAN << "\n종료합니다.\n" << RESET;
                return 0;
        }
    }
}

