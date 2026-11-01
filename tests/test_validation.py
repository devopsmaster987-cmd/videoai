"""Tests 26 + 28: voice availability detection and final output validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from freeanimate.validation import validate_output, write_report


def test_voice_availability_detection_reports_actual_voices():
    from freeanimate.tts import probe_all_engines
    reports = {r.engine: r for r in probe_all_engines()}
    # Voices are only listed when the engine actually probed available.
    for r in reports.values():
        if not r.available:
            assert r.voices == [], f"{r.engine} must not list voices when unavailable"


def test_voice_language_filtering():
    from freeanimate.tts import pick_engine
    eng = pick_engine("auto")
    voices = eng.list_voices("ur")
    # Whatever the result, every listed voice claims ur language prefix or engine is silence
    for v in voices:
        if eng.name != "silence":
            assert v.language.lower().startswith("ur") or v.language == "und"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_missing_file(tmp_path: Path):
    r = validate_output(tmp_path / "nope.mp4")
    assert r.ok is False
    assert any(c.name == "file_exists" and not c.ok for c in r.checks)


def test_validate_empty_file(tmp_path: Path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    r = validate_output(p)
    assert r.ok is False
    assert any(c.name == "file_not_empty" and not c.ok for c in r.checks)


def test_validate_non_media_file(tmp_path: Path):
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"not a media file " * 100)
    r = validate_output(p, require_audio=False)
    assert r.ok is False  # ffprobe cannot read it


def test_validate_real_mp4(tmp_path: Path, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    from freeanimate.ffmpeg_utils import run_ffmpeg
    p = tmp_path / "real.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=duration=2:size=640x360:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        "-pix_fmt", "yuv420p", str(p),
    ], timeout=300)
    r = validate_output(p, expected_width=640, expected_height=360,
                        expected_fps=24, requested_seconds=2.0,
                        duration_tolerance=0.2, require_audio=True)
    assert r.ok, r.summary()
    assert r.actual_duration == pytest.approx(2.0, abs=0.3)
    assert r.resolution == "640x360"
    assert r.audio_present is True
    assert r.frame_rate == pytest.approx(24, abs=0.5)


def test_validate_wrong_resolution_fails(tmp_path: Path, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    from freeanimate.ffmpeg_utils import run_ffmpeg
    p = tmp_path / "wrong.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=duration=1:size=640x360:rate=24",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(p),
    ], timeout=300)
    r = validate_output(p, expected_width=1920, expected_height=1080, require_audio=False)
    assert r.ok is False
    assert any(c.name == "resolution_matches" and not c.ok for c in r.checks)


def test_duration_tolerance_flags_but_reports(tmp_path: Path, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    from freeanimate.ffmpeg_utils import run_ffmpeg
    p = tmp_path / "short.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=duration=2:size=640x360:rate=24",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(p),
    ], timeout=300)
    # Ask for 60s but produce 2s -> outside tolerance
    r = validate_output(p, requested_seconds=60.0, duration_tolerance=0.15,
                        require_audio=False)
    assert any(c.name == "duration_within_tolerance" for c in r.checks)
    assert any("deviates" in lim for lim in r.limitations)


def test_subtitle_file_requirements(tmp_path: Path, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    from freeanimate.ffmpeg_utils import run_ffmpeg
    p = tmp_path / "withsubs.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=duration=1:size=640x360:rate=24",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(p),
    ], timeout=300)
    good = tmp_path / "s.srt"
    good.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    missing = tmp_path / "missing.srt"
    r = validate_output(p, require_audio=False, subtitle_files=[good])
    assert r.ok
    r2 = validate_output(p, require_audio=False, subtitle_files=[missing])
    assert r2.ok is False


def test_report_written(tmp_path: Path, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    from freeanimate.ffmpeg_utils import run_ffmpeg
    p = tmp_path / "v.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x240:rate=25",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(p),
    ], timeout=300)
    r = validate_output(p, expected_width=320, expected_height=240, require_audio=False)
    jp, tp = write_report(r, tmp_path / "reports")
    assert jp.exists() and tp.exists()
    assert "PASSED" in tp.read_text(encoding="utf-8") or "FAILED" in tp.read_text(encoding="utf-8")
