"""Tests 10: subtitle timestamp generation and validation."""

from __future__ import annotations

import pytest

from freeanimate.subtitles import (
    build_cues_for_scene,
    build_track,
    parse_srt,
    to_srt,
    to_vtt,
    wrap_lines,
)


def test_cue_timing_monotonic():
    cues = build_cues_for_scene(
        "First sentence here. Second sentence follows it. Third sentence ends the thought.",
        0.0, 12.0,
    )
    assert cues
    for a, b in zip(cues, cues[1:]):
        assert b.start >= a.start
        assert b.start >= a.end - 0.001 or b.start >= a.start
    assert cues[-1].end == pytest.approx(12.0, abs=0.2)


def test_cue_validity():
    cues = build_cues_for_scene("Hello world. More words follow here.", 1.0, 5.0)
    track = build_track([], "en")
    track.cues = cues
    assert track.validate() == []


def test_srt_roundtrip():
    cues = build_cues_for_scene("Line one is here. Line two too.", 0.0, 6.0)
    track = build_track([], "en")
    track.cues = cues
    srt = to_srt(track)
    parsed = parse_srt(srt)
    assert len(parsed) == len(cues)
    for orig, back in zip(cues, parsed):
        assert back.start == pytest.approx(orig.start, abs=0.001)
        assert back.end == pytest.approx(orig.end, abs=0.001)
        assert orig.text.split("\n")[0] in back.text


def test_vtt_format():
    cues = build_cues_for_scene("Hello there. Second cue text.", 0.0, 4.0)
    track = build_track([], "ur", rtl=True)
    track.cues = cues
    vtt = to_vtt(track)
    assert vtt.startswith("WEBVTT")
    assert "RTL" in vtt


def test_wrap_lines_limits():
    text = "word " * 40
    wrapped = wrap_lines(text, max_chars=46, max_lines=2)
    lines = wrapped.split("\n")
    assert len(lines) <= 2


def test_track_from_scenes_with_measured_audio():
    scenes = [
        {"scene_id": "scene_001", "duration_seconds": 10.0,
         "subtitle_text": "Narration for the first scene. It has two sentences."},
        {"scene_id": "scene_002", "duration_seconds": 8.0,
         "subtitle_text": "Narration for the second scene."},
    ]
    track = build_track(scenes, language="en", rtl=False,
                         narration_durations={"scene_001": 6.5, "scene_002": 5.0})
    assert track.validate() == []
    # Cues from scene 2 start after scene 1's window
    scene2_start = min(c.start for c in track.cues if c.start >= 9.0)
    assert scene2_start >= 9.0
    assert track.timing_source == "audio_measured"


def test_empty_scene_produces_no_cues():
    track = build_track([{"scene_id": "s", "duration_seconds": 5, "subtitle_text": ""}], "en")
    assert track.cues == []


def test_negative_or_zero_duration_detected():
    from freeanimate.subtitles import SubtitleCue, SubtitleTrack
    t = SubtitleTrack()
    t.cues = [SubtitleCue(1, 5.0, 4.0, "bad"), SubtitleCue(2, -1.0, 1.0, "neg")]
    problems = t.validate()
    assert any("end <= start" in p for p in problems)
    assert any("negative start" in p for p in problems)


def test_overlapping_cues_detected():
    from freeanimate.subtitles import SubtitleCue, SubtitleTrack
    t = SubtitleTrack()
    t.cues = [SubtitleCue(1, 0.0, 5.0, "a"), SubtitleCue(2, 3.0, 6.0, "b")]
    problems = t.validate()
    assert any("overlaps" in p for p in problems)


def test_per_scene_cues_are_local_and_distinct():
    """Regression: burning global SRT into every segment repeated scene 1 text."""
    from freeanimate.subtitles import build_scene_cues, to_ass
    scenes = [
        {"scene_id": "s1", "duration_seconds": 10.0, "subtitle_text": "First scene words here. More first words."},
        {"scene_id": "s2", "duration_seconds": 8.0, "subtitle_text": "Second scene text different. End of second."},
    ]
    durs = {"s1": 9.0, "s2": 7.0}
    c1 = build_scene_cues(scenes[0], durs)
    c2 = build_scene_cues(scenes[1], durs)
    assert c1 and c2
    assert c1[0].start < 1.0 and c2[0].start < 1.0  # local time base
    assert all(c.end <= 10.0 for c in c1)
    assert "First" in c1[0].text
    assert "First" not in c2[0].text
    ass1 = to_ass(c1, font_family="Nirmala UI", font_size=40, rtl=True)
    ass2 = to_ass(c2, font_family="Nirmala UI", font_size=40, rtl=True)
    assert "[Script Info]" in ass1 and "Style: Default,Nirmala UI,40" in ass1
    assert "Second" in ass2 and "First" not in ass2


def test_ass_rtl_marker_and_newlines():
    from freeanimate.subtitles import SubtitleCue, to_ass
    cues = [SubtitleCue(1, 0.0, 2.0, "line one\nline two")]
    ass = to_ass(cues, rtl=True)
    assert (chr(92) + "N") in ass
    assert "\u202b" in ass


def test_ass_time_format():
    from freeanimate.subtitles import _fmt_ass_time
    assert _fmt_ass_time(0) == "0:00:00.00"
    assert _fmt_ass_time(61.5) == "0:01:01.50"


def test_ass_title_overlay_and_family():
    from freeanimate.subtitles import SubtitleCue, to_ass
    ass = to_ass(
        [SubtitleCue(1, 0.0, 2.0, "sub")],
        font_family="Nirmala UI",
        title="کائنات کی شروعات",
        title_seconds=12.5,
        title_font_family="Nirmala UI",
        rtl=True,
    )
    assert "Style: Title,Nirmala UI," in ass
    assert "BorderStyle" not in ass.splitlines()[7]  # Default style untouched
    assert "Dialogue: 1,0:00:00.00,0:00:12.50,Title" in ass
    assert "\u202b" in ass  # RTL marker on title and subs
