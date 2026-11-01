"""Tests 11-14: FFmpeg command construction, FFprobe parsing, resolution
validation, and missing-FFmpeg handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from freeanimate.constants import RESOLUTIONS
from freeanimate.ffmpeg_utils import (
    FFmpegError,
    FFmpegNotFoundError,
    find_ffmpeg,
    has_filter,
    parse_rate_to_fps,
    probe,
    require_ffmpeg,
    run_ffmpeg,
)
from freeanimate.renderer import build_scene_command, concat_segments, zoompan_expressions
from freeanimate.visuals import MotionSpec


def test_zoompan_expressions_per_move():
    for move in ("push_in", "pull_out", "pan_left", "pan_right", "pan_up", "pan_down",
                 "drift_diagonal", "slow_zoom_center"):
        z, x, y = zoompan_expressions(move)
        assert "on" in z or "sin" in z or z.replace(".", "").isdigit() or "d" in z
        assert x and y


def test_scene_command_structure(tmp_path: Path):
    img = tmp_path / "scene.png"
    img.write_bytes(b"png-bytes")
    out = tmp_path / "out.mp4"
    motion = MotionSpec(camera_move="push_in", duration_seconds=5.0, fps=24,
                        input_w=2560, input_h=1440, out_w=1920, out_h=1080)
    cmd = build_scene_command(img, out, motion, crf=21, preset="veryfast")
    joined = " ".join(cmd)
    assert "-vf" in cmd
    assert "zoompan=" in joined
    assert "1920x1080" in joined
    assert str(out) in cmd
    assert "libx264" in cmd
    assert "-crf" in cmd
    # frames computed correctly
    fi = cmd.index("-frames:v")
    assert cmd[fi + 1] == str(5 * 24)


def test_scene_command_with_subtitles(tmp_path: Path):
    img = tmp_path / "scene.png"
    img.write_bytes(b"x")
    sub = tmp_path / "sub.srt"
    sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    motion = MotionSpec(camera_move="pan_right", duration_seconds=3, fps=24,
                        input_w=2560, input_h=1440, out_w=1920, out_h=1080)
    cmd = build_scene_command(img, tmp_path / "o.mp4", motion, subtitle_path=sub)
    joined = " ".join(cmd)
    assert "subtitles=" in joined
    # Windows path escaping: backslashes converted to forward slashes
    assert "\\" not in joined.split("subtitles=")[1].split("'")[1].replace("\\:", "")


def test_fades_in_command(tmp_path: Path):
    motion = MotionSpec(camera_move="push_in", duration_seconds=6, fps=24,
                        input_w=2560, input_h=1440, out_w=1920, out_h=1080)
    cmd = build_scene_command(tmp_path / "i.png", tmp_path / "o.mp4", motion,
                              fade_in=0.4, fade_out=0.4)
    joined = " ".join(cmd)
    assert "fade=t=in" in joined
    assert "fade=t=out:st=5.600" in joined


def test_concat_list_escaping(tmp_path: Path):
    sub = tmp_path / "with space"
    sub.mkdir()
    seg1 = sub / "a segment.mp4"
    seg1.write_bytes(b"x" * 2048)
    out = tmp_path / "final.mp4"
    if not find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    try:
        concat_segments([seg1], out)
    except FFmpegError:
        # Copy-concat of a non-media file legitimately fails; the important
        # assertion is that the list file escaped the path correctly.
        pass
    listfile = out.parent / "concat_list.txt"
    assert listfile.exists()
    content = listfile.read_text(encoding="utf-8")
    assert "with%20space" in content or "with space" in content
    assert content.startswith("file '")


def test_missing_ffmpeg_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("freeanimate.ffmpeg_utils.shutil.which", lambda *_: None)
    monkeypatch.setattr("freeanimate.ffmpeg_utils._candidate_dirs", lambda: [tmp_path / "nope"])
    with pytest.raises(FFmpegNotFoundError, match="FFmpeg"):
        require_ffmpeg("ffmpeg")


def test_run_ffmpeg_missing_binary_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("freeanimate.ffmpeg_utils.shutil.which", lambda *_: None)
    monkeypatch.setattr("freeanimate.ffmpeg_utils._candidate_dirs", lambda: [tmp_path / "nope"])
    with pytest.raises(FFmpegNotFoundError):
        run_ffmpeg(["-version"])


def test_probe_missing_file():
    info = probe(Path("definitely/not/here.mp4"))
    assert info.exists is False
    assert info.streams == []


def test_parse_rate():
    assert parse_rate_to_fps("30/1") == 30.0
    assert parse_rate_to_fps("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert parse_rate_to_fps("0/0") is None
    assert parse_rate_to_fps(None) is None


def test_resolution_validation():
    assert RESOLUTIONS["1080p"] == (1920, 1080)
    assert RESOLUTIONS["4k"] == (3840, 2160)
    for name, (w, h) in RESOLUTIONS.items():
        assert w % 2 == 0 and h % 2 == 0, f"{name} must be even-dimensioned for yuv420p"
        assert h < w


def test_has_filter_safe_when_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("freeanimate.ffmpeg_utils.shutil.which", lambda *_: None)
    monkeypatch.setattr("freeanimate.ffmpeg_utils._candidate_dirs", lambda: [tmp_path / "nope"])
    assert has_filter("zoompan") is False


@pytest.mark.skipif(find_ffmpeg("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_has_required_filters():
    for f in ("zoompan", "subtitles", "afade", "fade", "amix", "sidechaincompress"):
        assert has_filter(f), f"ffmpeg missing required filter: {f}"
