"""Tests 3-4: duration presets and custom duration validation."""

from __future__ import annotations

import pytest

from freeanimate.constants import (
    DURATION_CHOICES,
    MAX_CUSTOM_DURATION_MIN,
    MIN_CUSTOM_DURATION_MIN,
)


def test_all_predefined_choices_present():
    expected = {1: 5, 2: 10, 3: 15, 4: 20, 5: 25, 6: 30, 7: 40, 8: 45, 9: None}
    assert DURATION_CHOICES == expected


def test_predefined_minutes_in_required_set():
    values = {v for v in DURATION_CHOICES.values() if v is not None}
    assert values == {5, 10, 15, 20, 25, 30, 40, 45}


def test_custom_choice_is_option_9():
    assert DURATION_CHOICES[9] is None


def test_custom_duration_bounds():
    assert MIN_CUSTOM_DURATION_MIN == 1
    assert MAX_CUSTOM_DURATION_MIN >= 45


@pytest.mark.parametrize("minutes", [5, 10, 15, 20, 25, 30, 40, 45])
def test_duration_affects_word_budget(minutes):
    from freeanimate.script_engine import compute_word_budget
    w5 = compute_word_budget(5, 130)
    w45 = compute_word_budget(45, 130)
    words = compute_word_budget(minutes, 130)
    # Monotonic growth with duration
    if minutes == 45:
        assert words == w45
    assert words >= w5 * (minutes / 5) * 0.7
    # 45 min @ 130 wpm ≈ 5850 words (spec example), with pause ratio slightly under
    assert 5000 <= compute_word_budget(45, 130) <= 6300


def test_duration_validation_cli():
    """Invalid duration input is rejected (via pipeline options semantics)."""
    from freeanimate.pipeline import PipelineOptions
    opts = PipelineOptions(duration_minutes=10)
    assert opts.duration_minutes == 10
    # Non-positive durations must not survive validation anywhere in the app:
    with pytest.raises(Exception):
        from freeanimate.config import Config, ConfigError
        c = Config()
        c.set("defaults.duration_minutes", -5)


def test_duration_drives_scene_count():
    from freeanimate.script_engine import TemplateScriptEngine
    from freeanimate.scene_planner import ScenePlanner
    s5 = TemplateScriptEngine().generate("Explain black holes", target_minutes=5)
    s45 = TemplateScriptEngine().generate("Explain black holes", target_minutes=45)
    assert s45.actual_words > s5.actual_words * 5
    m5 = ScenePlanner(wpm=130, seed=1).plan(s5, 5 * 60)
    m45 = ScenePlanner(wpm=130, seed=1).plan(s45, 45 * 60)
    assert len(m45.scenes) > len(m5.scenes)
    # Long videos must not be a short video stretched:
    assert m45.total_estimated_seconds > m5.total_estimated_seconds * 3
