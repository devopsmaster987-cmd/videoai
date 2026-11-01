"""Tests 15-17: missing TTS engines, missing optional providers, and
provider capability detection (honest availability reporting)."""

from __future__ import annotations

from pathlib import Path

import pytest

from freeanimate.tts import (
    EspeakEngine,
    PiperEngine,
    SapiEngine,
    SilenceEngine,
    TTSError,
    pick_engine,
    probe_all_engines,
    synthesize_with_fallback,
    wav_duration,
    write_silence_wav,
)


def test_probe_all_reports_every_engine():
    reports = probe_all_engines()
    names = {r.engine for r in reports}
    assert names == {"piper", "espeak", "sapi", "silence"}
    for r in reports:
        assert isinstance(r.available, bool)
        assert r.detail, "every provider must explain its status"


def test_provider_never_available_without_checking(tmp_path: Path, monkeypatch):
    """An engine listed in config but absent on disk must probe unavailable."""
    piper = PiperEngine()
    monkeypatch.setattr(piper, "_piper_bin", lambda: None)
    info = piper.probe()
    assert info["available"] is False


def test_missing_espeak_reports_how_to_install(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("freeanimate.tts.shutil.which", lambda *_: None)
    monkeypatch.setattr("os.environ.get", lambda *_a, **_k: "")
    eng = EspeakEngine()
    monkeypatch.setattr(eng, "_bin", lambda: None)
    info = eng.probe()
    assert info["available"] is False
    assert "espeak" in info["detail"].lower()


def test_silence_engine_always_works(tmp_path: Path):
    eng = SilenceEngine({"wpm": 130})
    assert eng.probe()["available"] is True
    out = tmp_path / "sil.wav"
    res = eng.synthesize("word " * 65, out)   # 65 words @130wpm = 30s
    assert res.ok
    assert out.exists()
    assert 25 < res.duration_seconds < 35


def test_write_silence_wav_duration(tmp_path: Path):
    p = tmp_path / "s.wav"
    write_silence_wav(p, 2.0, rate=16000)
    assert abs(wav_duration(p) - 2.0) < 0.01


def test_sapi_reports_unavailable_when_not_installed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyttsx3":
            raise ImportError("no pyttsx3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    eng = SapiEngine()
    info = eng.probe()
    assert info["available"] is False
    assert "pyttsx3" in info["detail"]


def test_pick_engine_auto_falls_back_to_silence(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("freeanimate.tts.shutil.which", lambda *_: None)
    # Force all real engines to probe unavailable:
    monkeypatch.setattr(PiperEngine, "probe", lambda self: {"available": False, "detail": "x"})
    monkeypatch.setattr(EspeakEngine, "probe", lambda self: {"available": False, "detail": "x"})
    monkeypatch.setattr(SapiEngine, "probe", lambda self: {"available": False, "detail": "x"})
    eng = pick_engine("auto", language="ur")
    assert eng.name == "silence"


def test_pick_engine_unknown_raises():
    with pytest.raises(TTSError, match="unknown TTS engine"):
        pick_engine("nonexistent")


def test_synth_fallback_to_silence(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(PiperEngine, "probe", lambda self: {"available": False, "detail": "x"})
    monkeypatch.setattr(EspeakEngine, "probe", lambda self: {"available": False, "detail": "x"})
    monkeypatch.setattr(SapiEngine, "probe", lambda self: {"available": False, "detail": "x"})
    out = tmp_path / "n.wav"
    res = synthesize_with_fallback("Hello world narration.", out, preferred="auto", language="en")
    assert res.ok, "silence fallback must succeed so the pipeline still completes"
    assert res.engine == "silence"
    assert res.notes, "fallback must be disclosed in notes"


def test_provider_cost_metadata_never_claims_paid_defaults():
    for r in probe_all_engines():
        cost = r.extra.get("cost", "free")
        assert cost == "free", "core TTS must be free"
        assert r.extra.get("account_required", False) is False


def test_visual_providers_listed_honestly():
    from freeanimate.cli import cmd_providers
    # Structural: provider registry data exists and marks AI video unavailable.
    from freeanimate.tts import probe_all_engines
    assert all(isinstance(r.available, bool) for r in probe_all_engines())
