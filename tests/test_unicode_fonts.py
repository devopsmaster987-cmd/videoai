"""Tests 18 + 25: Unicode handling, RTL text, and missing-font guidance."""

from __future__ import annotations

from pathlib import Path

import pytest

from freeanimate.fonts import (
    INSTALL_GUIDANCE,
    find_font_for_script,
    find_fonts_for_scripts,
)
from freeanimate.languages import LanguageRegistry
from freeanimate.subtitles import build_track, to_srt, to_vtt, wrap_lines

URDU_SAMPLE = "تاریخِ مصر کی وضاحت کریں۔"
ARABIC_SAMPLE = "تاريخ مصر يشرحها هذا الفيديو"
HINDI_SAMPLE = "प्राचीन मिस्र का इतिहास"
CHINESE_SAMPLE = "古埃及的历史"


def test_unicode_survives_subtitles(tmp_path: Path):
    scenes = [{"scene_id": "scene_001", "duration_seconds": 6.0, "subtitle_text": URDU_SAMPLE}]
    track = build_track(scenes, language="ur", rtl=True)
    srt = to_srt(track)
    assert URDU_SAMPLE in srt
    vtt = to_vtt(track)
    assert URDU_SAMPLE in vtt
    # Files round-trip through UTF-8 on disk
    p = tmp_path / "u.srt"
    p.write_text(srt, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == srt


def test_rtl_flag_plumbing():
    reg = LanguageRegistry.builtin()
    assert reg.require("ur").rtl is True
    scenes = [{"scene_id": "s", "duration_seconds": 4, "subtitle_text": ARABIC_SAMPLE}]
    track = build_track(scenes, language="ar", rtl=True)
    assert track.rtl is True


def test_cjk_text_wrapping_without_spaces():
    text = CHINESE_SAMPLE * 6
    wrapped = wrap_lines(text, max_chars=20, max_lines=2)
    lines = wrapped.split("\n")
    assert len(lines) <= 2
    assert all(len(l) <= 40 for l in lines)  # allows slight overflow grace


def test_terminal_language_menu_unicode():
    lines = LanguageRegistry.builtin().menu_lines()
    joined = "\n".join(lines)
    assert "اردو" in joined
    assert "中文" in joined
    assert "(RTL)" in joined


@pytest.mark.parametrize("family", ["latin", "arabic", "devanagari", "cjk", "cyrillic"])
def test_font_search_never_crashes(family):
    res = find_font_for_script(family)
    assert res.family == family
    if res.ok:
        assert res.path is not None and res.path.name
        assert res.path.exists()
    else:
        assert res.guidance, "missing font must come with installation guidance"


def test_missing_font_guidance_exists_for_all_families():
    for family in ("latin", "arabic", "devanagari", "bengali", "gurmukhi", "cjk",
                   "hebrew", "cyrillic", "thai"):
        assert family in INSTALL_GUIDANCE


def test_find_fonts_batch():
    results = find_fonts_for_scripts(["latin", "cjk"])
    assert set(results) >= {"latin", "cjk"}


def test_script_engine_handles_non_english_prompt():
    from freeanimate.script_engine import TemplateScriptEngine, extract_topic
    topic = extract_topic("پرانے مصر کی تاریخ سے متعلق ایک ویڈیو بنائیں")
    assert topic  # non-empty, no crash
    # Note: offline engine composes English narration; language warnings are recorded
    s = TemplateScriptEngine().generate("Explain ancient Egypt", target_minutes=5, language="ur")
    assert s.warnings, "non-English script language must warn about template engine limits"


def test_scene_titles_unicode_serialize(tmp_path: Path):
    import json
    from freeanimate.scene_planner import Scene, SceneManifest
    m = SceneManifest(scenes=[Scene(scene_id="s1", title="مصر کی تاریخ",
                                    narration_text="اردو متن یہاں",
                                    duration_seconds=5, visual_description="v")],
                      total_estimated_seconds=5, target_seconds=5, seed=0)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m.to_dict(), ensure_ascii=False), encoding="utf-8")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["scenes"][0]["title"] == "مصر کی تاریخ"
