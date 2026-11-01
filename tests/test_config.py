"""Tests 1-2: configuration loading and invalid configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freeanimate.config import DEFAULT_CONFIG, Config, ConfigError, _deep_merge


def test_default_config_loads(tmp_path: Path):
    cfg = Config.load(tmp_path / "config.json")
    assert cfg.get("defaults.resolution") == "1080p"
    assert cfg.get("defaults.fps") == 24
    assert (tmp_path / "config.json").exists(), "default config file should be written"


def test_config_roundtrip(tmp_path: Path):
    p = tmp_path / "c.json"
    cfg = Config.load(p)
    cfg.set("defaults.duration_minutes", 25)
    cfg.save()
    cfg2 = Config.load(p)
    assert cfg2.get("defaults.duration_minutes") == 25


def test_partial_config_merges_with_defaults(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"defaults": {"resolution": "720p"}}), encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.get("defaults.resolution") == "720p"
    assert cfg.get("defaults.fps") == 24  # preserved default


def test_invalid_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        Config.load(p)


@pytest.mark.parametrize("section,value,match", [
    ("defaults.resolution", "1440p", "resolution"),
    ("defaults.fps", 120, "fps"),
    ("defaults.wpm", 10, "wpm"),
    ("defaults.duration_tolerance", 1.5, "tolerance"),
    ("render.crf", 99, "crf"),
    ("tts.speed", 5.0, "speed"),
    ("defaults.animation_mode", "unknown", "animation_mode"),
    ("defaults.transition", "wipe", "transition"),
    ("network.allow_translation_upload", "yes", "boolean"),
])
def test_invalid_values_rejected(tmp_path: Path, section, value, match):
    cfg = Config()
    with pytest.raises(ConfigError, match=match):
        cfg.set(section, value)


def test_get_dotted_and_missing():
    cfg = Config()
    assert cfg.get("render.crf") == 21
    assert cfg.get("nope.nope", "fallback") == "fallback"


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1, "c": 2}}
    merged = _deep_merge(base, {"a": {"b": 9}})
    assert base["a"]["b"] == 1
    assert merged["a"]["b"] == 9
    assert merged["a"]["c"] == 2


def test_missing_section_rejected(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"defaults": {}}), encoding="utf-8")
    # Missing sections are auto-filled from defaults by merge -> valid
    cfg = Config.load(p)
    cfg.validate()
    # But a corrupted in-memory config must fail
    bad = Config(data={"defaults": {}})
    with pytest.raises(ConfigError):
        bad.validate()
