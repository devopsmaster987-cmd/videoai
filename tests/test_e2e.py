"""Section 22: end-to-end acceptance test.

Builds a real video from a real prompt using the lightweight fallback
pipeline: script -> scenes -> narration (silent fallback if no TTS) ->
illustrations -> rendered MP4 -> FFprobe validation.

Skips only when FFmpeg is genuinely not installed (reported as skipped, never
as passed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freeanimate.ffmpeg_utils import find_ffmpeg, probe
from freeanimate.languages import LanguageSettings
from freeanimate.pipeline import Pipeline, PipelineOptions
from freeanimate.project import Project

pytestmark = pytest.mark.skipif(find_ffmpeg("ffmpeg") is None,
                                reason="ffmpeg not installed — cannot produce a real MP4")


def _run_pipeline(tmp_path: Path, *, duration: float = 1.0, language: str = "en",
                  resolution: str = "720p", burn: bool = True, music: str = "none",
                  tts_engine: str = "silence") -> tuple[Project, object]:
    root = tmp_path / "projects"
    root.mkdir(parents=True, exist_ok=True)
    prompt = ("Create a short cinematic documentary explaining the history of the universe, "
              "from the Big Bang to the formation of galaxies, with narration and subtitles.")
    settings = {
        "duration_minutes": duration, "resolution": resolution, "fps": 24,
        "animation_mode": "hybrid", "transition": "fade", "burn_subtitles": burn,
        "music": music, "tts_engine": tts_engine, "low_resource": True,
    }
    langs = LanguageSettings(project=language).as_dict()
    project = Project.create(root, "E2E Universe", prompt, langs, settings)

    from freeanimate.config import Config
    cfg = Config()
    cfg.set("paths.projects_dir", str(root))
    opts = PipelineOptions(
        duration_minutes=duration, resolution=resolution, fps=24,
        animation_mode="hybrid", transition="fade", wpm=130,
        burn_subtitles=burn, music=music, tts_engine=tts_engine,
        low_resource=True, crf=28, x264_preset="ultrafast",
    )
    result = Pipeline(project, opts).run(resume=False)
    return project, result


def test_end_to_end_produces_valid_mp4(tmp_path: Path):
    project, result = _run_pipeline(tmp_path, duration=1.0)
    assert result.ok, f"pipeline failed: {result.error}\n{result.notes}"
    assert result.output_path is not None
    out = Path(result.output_path)
    assert out.exists()
    assert out.stat().st_size > 50_000

    # FFprobe validation report
    report = result.report
    assert report is not None and report.ok, report.summary() if report else "no report"
    info = probe(out)
    assert info.primary_video() is not None
    assert info.primary_audio() is not None
    assert info.duration and info.duration > 0.5

    # Resolution matches request (720p)
    v = info.primary_video()
    assert (v.width, v.height) == (1280, 720)

    # Subtitles produced
    subs = list((project.root / "subtitles").glob("*.srt")) + list((project.root / "subtitles").glob("*.vtt"))
    assert len(subs) >= 2

    # Scene manifest + script exist
    assert (project.root / "scenes" / "scenes.json").exists()
    assert (project.root / "script" / "script.json").exists()

    # Validation report written
    assert (project.root / "reports" / "validation_report.json").exists()

    # Honest methods reporting
    methods = project.manifest.get("methods_used", {})
    assert methods.get("ai_video_used") is False  # no AI video claimed
    assert methods.get("animation_methods")
    data = json.loads((project.root / "reports" / "validation_report.json").read_text(encoding="utf-8"))
    assert data["ok"] is True


def test_end_to_end_with_music_and_4k_fails_gracefully_if_too_heavy(tmp_path: Path):
    """4K path: allowed but must either succeed or fail with an honest error."""
    project, result = _run_pipeline(tmp_path, duration=1.0, resolution="4k",
                                    music="procedural")
    if result.ok:
        info = probe(Path(result.output_path))
        v = info.primary_video()
        assert (v.width, v.height) == (3840, 2160)
        assert result.report.methods_used.get("music") == "procedural"
    else:
        # Must be an honest failure, not a fake success
        assert result.error


def test_resume_after_partial_run(tmp_path: Path):
    project, result = _run_pipeline(tmp_path, duration=1.0)
    assert result.ok
    # Second run resumes and reuses assets (should succeed again)
    from freeanimate.config import Config
    cfg = Config()
    opts = PipelineOptions(duration_minutes=1.0, resolution="720p", fps=24,
                           tts_engine="silence", low_resource=True,
                           crf=28, x264_preset="ultrafast")
    project2 = Project.load(project.root)
    result2 = Pipeline(project2, opts).run(resume=True)
    assert result2.ok, result2.error
    # Status exported after both runs
    assert project2.manifest["status"] == "exported"


def test_scene_segments_all_present(tmp_path: Path):
    project, result = _run_pipeline(tmp_path, duration=1.0)
    assert result.ok
    segs = list((project.root / "renders" / "segments").glob("*_segment.mp4"))
    scenes = json.loads((project.root / "scenes" / "scenes.json").read_text(encoding="utf-8"))
    assert len(segs) == len(scenes["scenes"])
