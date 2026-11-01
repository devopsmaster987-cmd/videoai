"""Tests 5-6: script + scene-manifest validation and scene-duration math."""

from __future__ import annotations

import pytest

from freeanimate.scene_planner import MANIFEST_NAME, Scene, SceneManifest, ScenePlanner
from freeanimate.script_engine import (
    TemplateScriptEngine,
    compute_word_budget,
    extract_topic,
    guess_kind,
    parse_user_script,
)


def test_topic_extraction():
    assert "ancient Egypt" in extract_topic("Explain the history of ancient Egypt.")
    assert extract_topic("Create a cinematic documentary about the Big Bang.") .startswith("Big Bang")
    assert extract_topic("") == "Untitled Topic"


def test_kind_guessing():
    assert guess_kind("Create a story about a brave knight") == "story"
    assert guess_kind("how to install python step by step") == "tutorial"
    assert guess_kind("top 10 programming languages") == "listicle"


def test_word_budget_math():
    assert compute_word_budget(1, 130) < compute_word_budget(2, 130)
    assert compute_word_budget(45, 130) == pytest.approx(45 * 130 * 0.92, abs=2)


def test_script_valid_and_serializable():
    s = TemplateScriptEngine().generate("Explain photosynthesis", target_minutes=5)
    assert s.sections
    assert s.actual_words > 100
    assert s.script_source == "template"  # honest: not an LLM
    d = s.to_dict()
    s2 = type(s).from_dict(d)
    assert s2.actual_words == s.actual_words
    assert len(s2.sections) == len(s.sections)
    assert s.stable_hash() == s2.stable_hash()


@pytest.mark.parametrize("minutes", [5, 10, 15, 20, 25, 30, 40, 45])
def test_script_scales_with_duration(minutes):
    s = TemplateScriptEngine().generate("The history of astronomy", target_minutes=minutes)
    assert s.actual_words >= compute_word_budget(minutes, 130) * 0.5
    est_minutes = s.actual_words / s.target_wpm
    assert est_minutes > minutes * 0.5


def test_scene_manifest_valid():
    s = TemplateScriptEngine().generate("Ancient Rome", target_minutes=5)
    m = ScenePlanner(wpm=130, seed=7).plan(s, 5 * 60)
    assert m.validate() == []
    assert MANIFEST_NAME == "scenes.json"
    assert m.scenes[0].scene_id == "scene_001"
    assert all(sc.duration_seconds > 0 for sc in m.scenes)
    assert all(sc.narration_text.strip() for sc in m.scenes)


def test_scene_camera_moves_vary():
    s = TemplateScriptEngine().generate("Ancient Rome", target_minutes=10)
    m = ScenePlanner(wpm=130, seed=3).plan(s, 10 * 60)
    moves = [sc.camera_move for sc in m.scenes]
    assert len(set(moves)) >= 3, "camera moves must not be identical everywhere"
    # No immediate repeats
    for a, b in zip(moves, moves[1:]):
        assert a != b


def test_scene_serialization_roundtrip():
    s = TemplateScriptEngine().generate("Ancient Rome", target_minutes=5)
    m = ScenePlanner(wpm=130, seed=1).plan(s, 5 * 60)
    m2 = SceneManifest.from_dict(m.to_dict())
    assert len(m2.scenes) == len(m.scenes)
    assert m2.scenes[0].camera_move == m.scenes[0].camera_move
    assert m2.validate() == []


def test_scene_validation_catches_problems():
    bad = SceneManifest(
        scenes=[
            Scene(scene_id="a", title="", narration_text="", duration_seconds=0,
                  visual_description=""),
            Scene(scene_id="a", title="dup", narration_text="x", duration_seconds=5,
                  visual_description=""),
        ],
        total_estimated_seconds=5, target_seconds=5, seed=0,
    )
    problems = bad.validate()
    assert any("duplicate" in p for p in problems)
    assert any("non-positive" in p for p in problems)
    assert any("empty" in p for p in problems)


def test_scene_duration_bounds():
    s = TemplateScriptEngine().generate("Volcanoes", target_minutes=15)
    m = ScenePlanner(wpm=130, seed=5).plan(s, 15 * 60)
    for sc in m.scenes:
        assert 4.0 <= sc.duration_seconds <= 45.0


def test_resync_durations_uses_real_audio():
    s = TemplateScriptEngine().generate("Volcanoes", target_minutes=5)
    m = ScenePlanner(wpm=130, seed=5).plan(s, 5 * 60)
    real = {sc.scene_id: 7.5 for sc in m.scenes}
    ScenePlanner.resync_durations(m, real, gap=0.35)
    for sc in m.scenes:
        assert sc.duration_seconds == pytest.approx(7.85, abs=0.01)
        assert sc.subtitle_end <= sc.duration_seconds
        assert sc.subtitle_end > sc.subtitle_start


def test_user_script_parser():
    text = "# Chapter One\n\nHello world narration.\n\n# Chapter Two\n\nMore narration here."
    script = parse_user_script(text, topic="T")
    assert script.script_source == "user"
    assert len(script.sections) == 2
    assert script.sections[0].title == "Chapter One"
    assert "Hello world" in script.sections[0].narration


def test_long_user_script_split():
    text = " ".join(f"Sentence number {i} about the topic." for i in range(200))
    script = parse_user_script(text, topic="T")
    assert len(script.sections) >= 2


def test_extension_narration_hits_word_target():
    from freeanimate.script_engine import compose_extension_narration
    text = compose_extension_narration("ancient Egypt", 60, seed_index=0)
    words = len(text.split())
    assert 50 <= words <= 100
    assert "ancient egypt" in text.lower()


def test_extension_narration_varies_between_seeds():
    from freeanimate.script_engine import compose_extension_narration
    a = compose_extension_narration("black holes", 40, seed_index=0)
    b = compose_extension_narration("black holes", 40, seed_index=5)
    assert a != b


def test_cosmos_prompt_selects_science_kind():
    from freeanimate.script_engine import TemplateScriptEngine, extract_topic, guess_kind
    p = ("Create a cinematic documentary explaining the history of the universe, "
         "from the Big Bang to the formation of galaxies.")
    assert guess_kind(p) == "science"
    assert extract_topic(p) == "history of the universe"
    sc = TemplateScriptEngine().generate(p, target_minutes=3)
    text = " ".join(s.narration for s in sc.sections).lower()
    assert "society" not in text
    assert "seeds of decline" not in text
    assert sc.sections[0].title.startswith("The History of the Universe")


def test_topic_title_case_and_prose_article():
    from freeanimate.script_engine import _prose_topic, _title_case
    assert _prose_topic("history of the universe") == "the history of the universe"
    assert _prose_topic("Big Bang") == "Big Bang"
    assert _title_case("history of the universe: a journey through the evidence") == \
        "History of the Universe: A Journey Through the Evidence"
