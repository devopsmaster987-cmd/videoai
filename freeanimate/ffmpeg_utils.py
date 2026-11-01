"""FFmpeg / FFprobe discovery, command execution, and metadata parsing.

Windows-first: handles paths with spaces via list-form subprocess calls (never
shell strings), surfaces stderr in errors, and detects missing binaries with
actionable guidance instead of cryptic crashes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


class FFmpegError(RuntimeError):
    pass


class FFmpegNotFoundError(FFmpegError):
    pass


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration: float | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class MediaInfo:
    path: Path
    exists: bool
    size_bytes: int
    format_name: str | None
    duration: float | None
    bit_rate: int | None
    streams: list[StreamInfo]

    @property
    def video_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.codec_type == "video"]

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.codec_type == "audio"]

    @property
    def subtitle_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.codec_type == "subtitle"]

    def primary_video(self) -> StreamInfo | None:
        return self.video_streams[0] if self.video_streams else None

    def primary_audio(self) -> StreamInfo | None:
        return self.audio_streams[0] if self.audio_streams else None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env in ("FFMPEG_DIR", "FFMPEG_HOME"):
        v = os.environ.get(env)
        if v:
            dirs.append(Path(v))
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    dirs.append(Path(pf) / "ffmpeg" / "bin")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    dirs.append(Path(pf86) / "ffmpeg" / "bin")
    chocolatey = os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey")
    dirs.append(Path(chocolatey) / "bin")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "WinGet" / "Links")
    return dirs


def find_ffmpeg(tool: str = "ffmpeg") -> str | None:
    """Locate ffmpeg/ffprobe; check PATH first, then common install dirs."""
    found = shutil.which(tool)
    if found:
        return found
    exe = f"{tool}.exe"
    found = shutil.which(exe)
    if found:
        return found
    for d in _candidate_dirs():
        for name in (exe, tool):
            p = d / name
            if p.is_file():
                return str(p)
    return None


def require_ffmpeg(tool: str = "ffmpeg") -> str:
    path = find_ffmpeg(tool)
    if not path:
        raise FFmpegNotFoundError(
            f"{tool} was not found. Install FFmpeg for Windows:\n"
            "  winget install Gyan.FFmpeg\n"
            "or download from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH.\n"
            "Then restart the terminal and run: freeanimate doctor"
        )
    return path


def require_ffprobe(tool: str = "ffprobe") -> str:
    """Like :func:`require_ffmpeg` but for ffprobe, with its own guidance."""
    path = find_ffmpeg(tool)
    if not path:
        raise FFmpegNotFoundError(
            f"{tool} was not found. It ships together with FFmpeg:\n"
            "  winget install Gyan.FFmpeg\n"
            "Then restart the terminal and run: freeanimate doctor"
        )
    return path


def ffmpeg_version() -> str | None:
    path = find_ffmpeg("ffmpeg")
    if not path:
        return None
    try:
        proc = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").splitlines()
    return first[0] if first else None


def ffprobe_version() -> str | None:
    path = find_ffprobe("ffprobe")
    if not path:
        return None
    try:
        proc = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").splitlines()
    return first[0] if first else None


def has_filter(name: str) -> bool:
    """Check that ffmpeg was built with a specific filter (e.g. zoompan)."""
    path = find_ffmpeg("ffmpeg")
    if not path:
        return False
    try:
        proc = subprocess.run([path, "-hide_banner", "-filters"], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    return bool(re.search(rf"\b{re.escape(name)}\b", proc.stdout or ""))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_ffmpeg(args: Sequence[str], *, timeout: int = 3600, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run ffmpeg with a list of args (safe for paths with spaces)."""
    binary = require_ffmpeg("ffmpeg")
    cmd = [binary, "-hide_banner", "-y", "-nostdin", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg timed out after {timeout}s: {' '.join(cmd[:8])}...") from exc
    except OSError as exc:
        raise FFmpegError(f"failed to launch ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-25:])
        raise FFmpegError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")
    return proc


def run_ffprobe(args: Sequence[str], *, timeout: int = 60) -> str:
    binary = require_ffprobe("ffprobe")
    cmd = [binary, "-hide_banner", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffprobe timed out after {timeout}s") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-15:])
        raise FFmpegError(f"ffprobe failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout or ""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "N/A"):
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d else None
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def probe(path: Path | str) -> MediaInfo:
    """Inspect a media file with ffprobe; raises on unreadable files."""
    p = Path(path)
    if not p.exists():
        return MediaInfo(p, False, 0, None, None, None, [])
    out = run_ffprobe([
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(p),
    ])
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {p}: {exc}") from exc

    fmt = data.get("format", {}) or {}
    streams: list[StreamInfo] = []
    for s in data.get("streams", []) or []:
        streams.append(
            StreamInfo(
                index=int(s.get("index", 0)),
                codec_type=str(s.get("codec_type", "unknown")),
                codec_name=s.get("codec_name"),
                width=s.get("width"),
                height=s.get("height"),
                avg_frame_rate=s.get("avg_frame_rate") or s.get("r_frame_rate"),
                sample_rate=int(s["sample_rate"]) if s.get("sample_rate") else None,
                channels=s.get("channels"),
                duration=float(s["duration"]) if s.get("duration") else None,
                tags={str(k): str(v) for k, v in (s.get("tags") or {}).items()},
            )
        )
    duration = None
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except ValueError:
            duration = None
    bit_rate = None
    if fmt.get("bit_rate"):
        try:
            bit_rate = int(fmt["bit_rate"])
        except ValueError:
            bit_rate = None
    return MediaInfo(
        path=p,
        exists=True,
        size_bytes=p.stat().st_size,
        format_name=fmt.get("format_name"),
        duration=duration,
        bit_rate=bit_rate,
        streams=streams,
    )


def parse_rate_to_fps(rate: str | None) -> float | None:
    return _parse_rate(rate)
