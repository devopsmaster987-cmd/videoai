

def test_auto_engine_never_uses_mismatched_voice(monkeypatch):
    """Regression: auto-pick kept SAPI for Urdu (English-only voices)."""
    from freeanimate import tts as T

    class _SapiLike:
        name = "sapi"

        def __init__(self, cfg=None):
            pass

        def probe(self):
            return {"available": True}

        def list_voices(self, language=None):
            # English-only voice bank, filtered like the real engines.
            voices = [T.VoiceInfo(id="v1", name="David", language="en-US",
                                  engine="sapi", offline=True)]
            if language:
                voices = [v for v in voices
                          if T._normalize_lang(v.language) == T._normalize_lang(language)]
            return voices

    class _Unavailable:
        def __init__(self, name):
            self.name = name

        def probe(self):
            return {"available": False}

    def _unav(name):
        def factory(cfg=None):
            return _Unavailable(name)
        return factory

    monkeypatch.setitem(T._ENGINE_CLASSES, "sapi", _SapiLike)
    monkeypatch.setitem(T._ENGINE_CLASSES, "piper", _unav("piper"))
    monkeypatch.setitem(T._ENGINE_CLASSES, "espeak", _unav("espeak"))
    eng = T.pick_engine("auto", language="ur")
    assert eng.name == "silence"
    eng = T.pick_engine("auto", language="en")
    assert eng.name == "sapi"
    # Explicit user override still wins.
    eng = T.pick_engine("sapi", language="ur")
    assert eng.name == "sapi"
