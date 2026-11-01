"""Final output validation (section 18).

After rendering, the validator uses FFprobe to confirm the file exists, is
non-empty, decodable, has the expected streams, resolution, duration within
tolerance, and that requested subtitle files exist.  It never claims success
because FFmpeg was merely launched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .ffmpeg_utils import MediaInfo, probe, run_ffprobe, FFmpegError


@dataclass
class ValidationCheck:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "fail"   # "fail" | "warn" | "info"

    def label(self) -> str:
        if self.ok:
            return "OK "
        if self.severity == "warn":
            return "WARN"
        if self.severity == "info":
            return "INFO"
        return "FAIL"


@dataclass
class ValidationReport:
    output_path: str
    ok: bool
    checks: list[ValidationCheck] = field(default_factory=list)
    media: dict[str, Any] | None = None
    actual_duration: float | None = None
    requested_duration: float | None = None
    resolution: str | None = None
    frame_rate: float | None = None
    audio_present: bool = False
    subtitle_files: list[str] = field(default_factory=list)
    methods_used: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def add(self, name: str, ok: bool, detail: str = "", severity: str = "fail") -> None:
        self.checks.append(ValidationCheck(name, ok, detail, severity))
        if not ok and severity == "fail":
            self.ok = False

    @property
    def warnings(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]

    def summary(self) -> str:
        lines = [f"Validation: {'PASSED' if self.ok else 'FAILED'}"]
        for c in self.checks:
            lines.append(f"  [{c.label()}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


def validate_output(
    output_path: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: float | None = None,
    requested_seconds: float | None = None,
    duration_tolerance: float = 0.15,
    require_audio: bool = True,
    subtitle_files: list[Path] | None = None,
    expected_scenes: int | None = None,
    rendered_segments: int | None = None,
    methods_used: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> ValidationReport:
    report = ValidationReport(output_path=str(output_path), ok=True,
                              methods_used=methods_used or {},
                              limitations=list(limitations or []))

    # 1. Existence + non-empty
    exists = output_path.exists()
    report.add("file_exists", exists, str(output_path))
    if not exists:
        report.ok = False
        return report
    size = output_path.stat().st_size
    report.add("file_not_empty", size > 1024, f"{size:,} bytes")
    if size <= 1024:
        report.ok = False
        return report

    # 2. Readable container + streams
    try:
        info: MediaInfo = probe(output_path)
    except FFmpegError as exc:
        report.add("container_readable", False, str(exc))
        report.ok = False
        return report
    report.add("container_readable", True, info.format_name or "unknown")

    v = info.primary_video()
    report.add("video_stream_present", v is not None,
               f"codec={v.codec_name}" if v else "no video stream")
    if v is None:
        report.ok = False

    a = info.primary_audio()
    report.audio_present = a is not None
    if require_audio:
        report.add("audio_stream_present", a is not None,
                   f"codec={a.codec_name}" if a else "no audio stream")
        if a is None:
            report.ok = False

    # 3. Resolution
    if v is not None:
        report.resolution = f"{v.width}x{v.height}"
        if expected_width and expected_height:
            match = (v.width == expected_width and v.height == expected_height)
            report.add("resolution_matches", match,
                       f"expected {expected_width}x{expected_height}, got {v.width}x{v.height}")
            if not match:
                report.ok = False
        # Frame rate
        from .ffmpeg_utils import parse_rate_to_fps
        fps = parse_rate_to_fps(v.avg_frame_rate)
        report.frame_rate = fps
        if expected_fps and fps:
            fps_ok = abs(fps - expected_fps) <= 0.5
            report.add("frame_rate_matches", fps_ok, f"expected ~{expected_fps}, got {fps:.3f}")
            if not fps_ok:
                report.ok = False

    # 4. Duration
    dur = info.duration
    report.actual_duration = dur
    report.requested_duration = requested_seconds
    if dur is None:
        report.add("duration_readable", False, "ffprobe returned no duration")
        report.ok = False
    else:
        report.add("duration_readable", True, f"{dur:.2f}s")
        if requested_seconds and requested_seconds > 0:
            delta = abs(dur - requested_seconds) / requested_seconds
            within = delta <= duration_tolerance
            report.add("duration_within_tolerance", within,
                       f"requested {requested_seconds:.1f}s, actual {dur:.1f}s "
                       f"({delta * 100:.1f}% off, tolerance {duration_tolerance * 100:.0f}%)",
                       severity="warn")
            if not within:
                # Warning-level: does not fail validation, but flagged.
                report.limitations.append(
                    f"Final duration {dur:.1f}s deviates {delta * 100:.1f}% from requested "
                    f"{requested_seconds:.1f}s (tolerance {duration_tolerance * 100:.0f}%)."
                )

    # 5. Decode check (read packets from the start; cheap smoke test)
    try:
        run_ffprobe(["-v", "error", "-read_intervals", "%+#3",
                     "-show_entries", "packet=codec_type", "-of", "csv", str(output_path)])
        report.add("decodable", True, "ffprobe read packets successfully")
    except FFmpegError as exc:
        report.add("decodable", False, str(exc)[:200])
        report.ok = False

    # 6. Subtitle files
    for sp in subtitle_files or []:
        ok = sp.exists() and sp.stat().st_size > 0
        report.subtitle_files.append(str(sp))
        report.add(f"subtitle_file:{sp.name}", ok, str(sp))
        if not ok:
            report.ok = False

    # 7. Scene coverage (informational)
    if expected_scenes and rendered_segments is not None:
        ok = rendered_segments >= expected_scenes
        report.add("all_scenes_rendered", ok,
                   f"{rendered_segments}/{expected_scenes} segments")
        if not ok:
            report.ok = False

    report.media = {
        "format_name": info.format_name,
        "size_bytes": info.size_bytes,
        "bit_rate": info.bit_rate,
        "duration": info.duration,
        "video": asdict(v) if v else None,
        "audio": asdict(a) if a else None,
    }
    if not report.ok:
        report.limitations.append("Validation reported failures; see checks for details.")
    return report


def write_report(report: ValidationReport, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    jp = reports_dir / "validation_report.json"
    tp = reports_dir / "validation_report.txt"
    jp.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tp.write_text(report.summary(), encoding="utf-8")
    return jp, tp
