"""Tests 18-21 + 24: language selection, custom language, unsupported
language handling, separate role languages, RTL flags."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freeanimate.languages import (
    BUILTIN_LANGUAGES,
    ROLE_NAMES,
    LanguageRegistry,
    LanguageSettings,
)


def test_menu_covers_required_languages():
    reg = LanguageRegistry.builtin()
    required = {"English", "Urdu", "Hindi", "Arabic", "Spanish", "French", "German",
                "Portuguese", "Chinese", "Japanese", "Korean", "Turkish", "Russian",
                "Bengali", "Punjabi", "Italian", "Indonesian"}
    names = {l.name for l in reg}
    assert required <= names
    assert len(reg) >= 19


def test_rtl_languages_flagged():
    reg = LanguageRegistry.builtin()
    for code in ("ur", "ar", "fa", "he"):
        assert reg.require(code).rtl is True, f"{code} must be RTL"
    for code in ("en", "fr", "hi", "zh"):
        assert reg.require(code).rtl is False


def test_resolve_by_code_name_and_native():
    reg = LanguageRegistry.builtin()
    assert reg.resolve("ur").native == "اردو"
    assert reg.resolve("Urdu").code == "ur"
    assert reg.resolve("اردو").code == "ur"
    assert reg.resolve("spanish").code == "es"


def test_unsupported_language_raises():
    reg = LanguageRegistry.builtin()
    with pytest.raises(KeyError, match="unsupported language"):
        reg.resolve("klingon")


def test_custom_language_from_file(tmp_path: Path):
    p = tmp_path / "languages.json"
    data = {"languages": [
        {"code": "xx", "name": "Testish", "native": "Testish", "rtl": False,
         "script": "latin", "font_hints": ["arial.ttf"], "tts": ["espeak"]},
        {"code": "yy", "name": "Broken", "rtl": "not-a-bool"},   # missing name -> invalid
    ]}
    # entry #1 lacks "name"? No — entry 2 lacks "name" key entirely:
    data["languages"][1] = {"code": "yy", "rtl": True}
    p.write_text(json.dumps(data), encoding="utf-8")
    reg, errors = LanguageRegistry.load_with_errors(p)
    assert "xx" in reg
    assert "yy" not in reg
    assert errors, "malformed entry must be reported"


def test_malformed_language_file_falls_back(tmp_path: Path):
    p = tmp_path / "languages.json"
    p.write_text("{broken", encoding="utf-8")
    reg, errors = LanguageRegistry.load_with_errors(p)
    assert len(reg) == len(BUILTIN_LANGUAGES)
    assert errors


def test_default_roles_all_follow_project_language():
    s = LanguageSettings(project="ur")
    assert s.resolved("script") == "ur"
    assert s.resolved("narration") == "ur"
    assert s.resolved("subtitle") == "ur"
    assert s.resolved("onscreen") == "ur"
    assert not s.uses_translation()


def test_independent_role_override():
    s = LanguageSettings(project="ur", subtitle="en", narration="hi")
    assert s.resolved("project") == "ur"
    assert s.resolved("subtitle") == "en"
    assert s.resolved("narration") == "hi"
    assert s.resolved("script") == "ur"
    assert s.uses_translation()
    d = s.as_dict()
    assert d["subtitle"] == "en" and d["narration"] == "hi"


def test_settings_roundtrip_dict():
    s = LanguageSettings(project="en", script="en", narration="hi", subtitle="en", onscreen="en")
    s2 = LanguageSettings.from_dict(s.as_dict())
    assert s2.resolved("narration") == "hi"
    assert s2.resolved("subtitle") == "en"


def test_unknown_role_rejected():
    s = LanguageSettings(project="en")
    with pytest.raises(KeyError):
        s.resolved("nonsense")


def test_all_roles_have_names():
    assert set(ROLE_NAMES) == {"project", "script", "narration", "subtitle",
                               "onscreen", "thumbnail"}


def test_language_file_save_load_cycle(tmp_path: Path):
    reg = LanguageRegistry.builtin()
    p = tmp_path / "langs.json"
    reg.save(p)
    reg2 = LanguageRegistry.from_file(p)
    assert reg2.codes == reg.codes
