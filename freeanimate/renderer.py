"""Scene renderer: still image + motion spec -> MP4 segment.

Uses FFmpeg ``zoompan`` for Ken Burns camera motion (push in, pull out, pans,
diagonal drift) with per-move expressions, plus optional burned-in subtitles
via the ``subtitles`` filter (libass: HarfBuzz shaping + FriBidi bidi, so
Urdu/Arabic render correctly).

Every camera move is expressed in terms of the *output frame number* so motion
is smooth (the classic zoompan stutter from using input timestamps is avoided).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ffmpeg_utils import FFmpegError, has_filter, run_ffmpeg
from .visuals import MotionSpec


@dataclass
class RenderResult:
    scene_id: str
    ok: bool
    segment_path: Path | None
    duration_seconds: float
    error: str | None = None
    method: str = "zoompan"


# ---------------------------------------------------------------------------
# zoompan expression library
# ---------------------------------------------------------------------------
def zoompan_expressions(move: str, max_zoom: float = 1.18, min_zoom: float = 1.02,
                        frames: int = 100) -> tuple[str, str, str]:
    """Return (zoom, x, y) expressions for the zoompan filter.

    ``on`` is the output frame index.  The total frame count is embedded as a
    literal because zoompan expressions do NOT expose ``d`` as a variable
    (doing so fails with "Undefined constant ... in 'd)'").
    """
    total = max(1, int(frames))
    if move == "push_in":
        z = f"{min_zoom}+({max_zoom}-{min_zoom})*on/{total}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif move == "pull_out":
        z = f"{max_zoom}-({max_zoom}-{min_zoom})*on/{total}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif move == "slow_zoom_center":
        mid = (max_zoom + min_zoom) / 2
        z = f"{mid}+({max_zoom}-{mid})*sin(PI*on/{total})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif move == "pan_right":
        z = f"{max_zoom}"
        x = f"(iw-iw/zoom)*on/{total}"
        y = "ih/2-(ih/zoom/2)"
    elif move == "pan_left":
        z = f"{max_zoom}"
        x = f"(iw-iw/zoom)*(1-on/{total})"
        y = "ih/2-(ih/zoom/2)"
    elif move == "pan_down":
        z = f"{max_zoom}"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*on/{total}"
    elif move == "pan_up":
        z = f"{max_zoom}"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1-on/{total})"
    elif move == "drift_diagonal":
        z = f"{min_zoom}+({max_zoom}-{min_zoom})*on/{total}"
        x = f"(iw-iw/zoom)*on/{total}"
        y = f"(ih-ih/zoom)*on/{total}"
    else:  # unknown -> gentle push in
        z = f"{min_zoom}+({max_zoom}-{min_zoom})*on/{total}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    return z, x, y


def build_scene_command(
    image_path: Path,
    out_path: Path,
    motion: MotionSpec,
    *,
    subtitle_path: Path | None = None,
    video_codec: str = "libx264",
    crf: int = 21,
    preset: str = "veryfast",
    pixel_format: str = "yuv420p",
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> list[str]:
    """Build the ffmpeg argument list for one scene segment."""
    frames = max(1, int(round(motion.duration_seconds * motion.fps)))
    z, x, y = zoompan_expressions(motion.camera_move, motion.max_zoom, motion.min_zoom, frames)

    # zoompan: upscale input frames first for quality, then apply motion.
    vf_parts = [
        f"scale={motion.input_w * 2}:{motion.input_h * 2}:flags=lanczos",
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={motion.out_w}x{motion.out_h}:fps={motion.fps}",
    ]
    if subtitle_path is not None and subtitle_path.exists():
        # libass handles shaping/bidi; escape Windows path for the filter graph.
        sub = _escape_filter_path(subtitle_path)
        vf_parts.append(f"subtitles='{sub}'")
    if fade_in > 0:
        vf_parts.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        st = max(0.0, motion.duration_seconds - fade_out)
        vf_parts.append(f"fade=t=out:st={st:.3f}:d={fade_out:.3f}")
    vf_parts.append(f"format={pixel_format}")

    vf = ",".join(vf_parts)
    return [
        "-loop", "1",
        "-framerate", str(motion.fps),
        "-i", str(image_path),
        "-vf", vf,
        "-frames:v", str(frames),
        "-c:v", video_codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", pixel_format,
        "-video_track_timescale", "90000",
        "-an",
        str(out_path),
    ]


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter string on Windows."""
    s = path.as_posix()
    s = s.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return s


def render_scene_segment(
    image_path: Path,
    out_path: Path,
    motion: MotionSpec,
    *,
    subtitle_path: Path | None = None,
    video_codec: str = "libx264",
    crf: int = 21,
    preset: str = "veryfast",
    pixel_format: str = "yuv420p",
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    timeout: int = 1200,
) -> RenderResult:
    scene_id = out_path.stem.replace("_segment", "")
    if not image_path.exists():
        return RenderResult(scene_id, False, None, 0.0, error=f"missing image: {image_path}")
    if not has_filter("zoompan"):
        return RenderResult(scene_id, False, None, 0.0,
                            error="ffmpeg build lacks the zoompan filter; install a full ffmpeg build")
    if subtitle_path is not None and subtitle_path.exists() and not has_filter("subtitles"):
        return RenderResult(scene_id, False, None, 0.0,
                            error="ffmpeg build lacks the subtitles (libass) filter required for burn-in")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = build_scene_command(
        image_path, out_path, motion,
        subtitle_path=subtitle_path,
        video_codec=video_codec, crf=crf, preset=preset, pixel_format=pixel_format,
        fade_in=fade_in, fade_out=fade_out,
    )
    try:
        run_ffmpeg(args, timeout=timeout)
    except FFmpegError as exc:
        return RenderResult(scene_id, False, None, 0.0, error=str(exc))
    if not out_path.exists() or out_path.stat().st_size == 0:
        return RenderResult(scene_id, False, None, 0.0, error="segment file missing or empty after render")
    return RenderResult(scene_id, True, out_path, motion.duration_seconds)


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------
def concat_segments(segment_paths: list[Path], out_path: Path, *, list_file: Path | None = None) -> Path:
    """Concatenate segments with the concat demuxer (stream copy).

    All segments must share codec parameters (the renderer guarantees this).
    Falls back to a re-encode pass if the copy concat produces an unreadable
    file (checked by the caller via ffprobe).
    """
    if not segment_paths:
        raise FFmpegError("no segments to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf = list_file or (out_path.parent / "concat_list.txt")
    lines = []
    for p in segment_paths:
        escaped = str(p.resolve()).replace("\\", "/").replace("'", r"\'")
        lines.append(f"file '{escaped}'")
    lf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", str(lf),
        "-c", "copy",
        str(out_path),
    ], timeout=3600)
    return out_path


def concat_segments_reencode(segment_paths: list[Path], out_path: Path, *,
                             fps: int, width: int, height: int,
                             crf: int = 21, preset: str = "veryfast",
                             pixel_format: str = "yuv420p") -> Path:
    """Robust fallback: re-encode while concatenating (slower, always works)."""
    if not segment_paths:
        raise FFmpegError("no segments to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    for p in segment_paths:
        args += ["-i", str(p)]
    n = len(segment_paths)
    if n == 1:
        fc = "[0:v]null[v]"
    else:
        parts = "".join(f"[{i}:v]" for i in range(n))
        fc = f"{parts}concat=n={n}:v=1:a=0[v]"
    args += [
        "-filter_complex", fc,
        "-map", "[v]",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", pixel_format,
        "-r", str(fps),
        "-video_track_timescale", "90000",
        str(out_path),
    ]
    run_ffmpeg(args, timeout=7200)
    return out_path
