import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import cv2
import ffmpeg


@dataclass
class VideoInfo:
    format: str = ""
    duration_sec: float = 0.0
    bitrate_bps: int = 0
    video_codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    nb_frames: int = 0
    audio_codec: str = ""
    sample_rate: str = ""
    channels: int = 0
    error: str = ""

    def is_valid(self) -> bool:
        return self.error == ""


def get_video_info(video_path: Path) -> VideoInfo:
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as e:
        return VideoInfo(error=str(e))

    info = VideoInfo()
    fmt = probe.get("format", {})
    info.format = fmt.get("format_long_name", "")
    info.duration_sec = float(fmt.get("duration", 0))
    info.bitrate_bps = int(fmt.get("bit_rate", 0))

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            info.video_codec = stream.get("codec_name", "")
            info.width = stream.get("width", 0)
            info.height = stream.get("height", 0)
            r = stream.get("r_frame_rate", "0/1").split("/")
            if len(r) == 2 and int(r[1]) != 0:
                info.fps = int(r[0]) / int(r[1])
            raw_frames = stream.get("nb_frames", "")
            if raw_frames.isdigit():
                info.nb_frames = int(raw_frames)
        elif stream.get("codec_type") == "audio":
            info.audio_codec = stream.get("codec_name", "")
            info.sample_rate = stream.get("sample_rate", "")
            info.channels = stream.get("channels", 0)

    return info


def convert_to_mp4(
    input_path: Path,
    output_path: Path,
    progress_cb: Callable[[int], None],
) -> None:
    """Convert any video to MP4 (H.264 + AAC). Raises RuntimeError on failure."""
    info = get_video_info(input_path)
    total_duration = info.duration_sec

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-progress", "pipe:1",
        "-nostats",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                if total_duration > 0:
                    pct = min(int(ms / 1_000_000 / total_duration * 100), 99)
                    progress_cb(pct)
            except (ValueError, IndexError):
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    progress_cb(100)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    ratio: int,
    grayscale: bool,
    progress_cb: Callable[[int, int], None],
) -> int:
    """Extract frames from video. ratio=1 → all, 2 → every other, 3 → every third.
    Returns number of saved frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % ratio == 0:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1
        if total > 0:
            progress_cb(frame_idx, total)

    cap.release()
    return saved
