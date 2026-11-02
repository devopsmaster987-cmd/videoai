"""Pluggable text-to-speech system (section 5).

Engines are probed at runtime and report *actual* availability — a provider is
never marked available just because it appears in configuration (section 9).

Built-in engines, in preference order for ``auto``:

1. **piper**  — offline neural TTS, best quality when a voice model is present.
2. **espeak** — offline formant TTS, wide language coverage, robotic voice.
3. **sapi**   — Windows Speech API via ``pyttsx3``; uses installed system voices.
4. **silence**— last-resort engine that emits correctly-timed silence so the
                pipeline still produces a real video (reported honestly).

Every engine returns PCM WAV data; narration length is measured from the real
audio so subtitle and scene timing stay accurate.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .constants import DEFAULT_SAMPLE_RATE

# ---------------------------------------------------------------------------
@dataclass
class VoiceInfo:
    id: str
    name: str
    language: str            # BCP-47-ish code, e.g. "en-US"
    engine: str
    offline: bool = True
    size_note: str = ""
    license_note: str = ""
    sample_rate: int = 22050

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class TTSResult:
    ok: bool
    audio_path: Path | None
    duration_seconds: float
    engine: str
    voice_id: str | None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class TTSError(RuntimeError):
    pass


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate) if rate else 0.0


def _normalize_lang(code: str) -> str:
    return code.replace("_", "-").split("-")[0].lower()


def _estimate_silence_seconds(text: str, wpm: int) -> float:
    words = max(1, len(text.split()))
    return words / float(wpm) * 60.0


def write_silence_wav(path: Path, seconds: float, rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    n = int(seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)
    return path


# ---------------------------------------------------------------------------
# Engine base + registry
# ---------------------------------------------------------------------------
class TTSEngine(ABC):
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        """Return {available: bool, detail: str, ...}. Must not raise."""

    @abstractmethod
    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        ...

    @abstractmethod
    def synthesize(self, text: str, out_path: Path, *, voice_id: str | None = None,
                   speed: float = 1.0, rate: int = DEFAULT_SAMPLE_RATE) -> TTSResult:
        ...

    def preview(self, text: str, out_path: Path, *, voice_id: str | None = None,
                speed: float = 1.0) -> TTSResult:
        short = text if len(text) <= 200 else text[:200]
        return self.synthesize(short, out_path, voice_id=voice_id, speed=speed)


class PiperEngine(TTSEngine):
    """Piper neural TTS — offline, high quality, needs a downloaded voice."""

    name = "piper"

    def _piper_bin(self) -> str | None:
        for cand in ("piper", "piper.exe"):
            found = shutil.which(cand)
            if found:
                return found
        # Common user-local install locations on Windows.
        local = os.environ.get("LOCALAPPDATA", "")
        for sub in ("piper", "piper-tts", "Programs\\piper"):
            p = Path(local) / sub / "piper.exe"
            if p.is_file():
                return str(p)
        return None

    def _voices_dir(self) -> Path | None:
        env = os.environ.get("PIPER_VOICES_DIR")
        if env and Path(env).is_dir():
            return Path(env)
        local = os.environ.get("LOCALAPPDATA", "")
        cand = Path(local) / "piper" / "voices"
        if cand.is_dir():
            return cand
        return None

    def probe(self) -> dict[str, Any]:
        binary = self._piper_bin()
        vdir = self._voices_dir()
        models: list[Path] = []
        if vdir:
            models = list(vdir.rglob("*.onnx"))
        ok = binary is not None and bool(models)
        detail_parts = []
        if binary is None:
            detail_parts.append("piper binary not found on PATH")
        if not models:
            detail_parts.append("no voice .onnx models found")
        return {
            "available": ok,
            "detail": "; ".join(detail_parts) if detail_parts else f"{len(models)} voice model(s) found",
            "binary": binary,
            "voices_dir": str(vdir) if vdir else None,
            "model_count": len(models),
            "cost": "free",
            "account_required": False,
            "licensing": "Piper is MIT-licensed; individual voices have their own licenses (check each voice).",
        }

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        info = self.probe()
        vdir = info.get("voices_dir")
        out: list[VoiceInfo] = []
        if not vdir:
            return out
        for model in Path(vdir).rglob("*.onnx"):
            stem = model.stem
            lang_guess = stem.split("-")[0] if "-" in stem else "unknown"
            if language and _normalize_lang(lang_guess) != _normalize_lang(language):
                continue
            out.append(
                VoiceInfo(
                    id=str(model),
                    name=stem,
                    language=lang_guess,
                    engine=self.name,
                    offline=True,
                    size_note=f"{model.stat().st_size / 1e6:.0f} MB model" if model.exists() else "",
                    license_note="See the voice's accompanying LICENSE file.",
                )
            )
        return out

    def synthesize(self, text, out_path, *, voice_id=None, speed=1.0, rate=DEFAULT_SAMPLE_RATE) -> TTSResult:
        info = self.probe()
        if not info["available"]:
            return TTSResult(False, None, 0.0, self.name, voice_id, error=info["detail"])
        binary = info["binary"]
        voices = self.list_voices()
        if not voices:
            return TTSResult(False, None, 0.0, self.name, voice_id, error="no piper voices installed")
        model = voice_id or voices[0].id
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wav_tmp = out_path.with_suffix(".piper.wav")
        cmd = [binary, "--model", model, "--output_file", str(wav_tmp)]
        if speed != 1.0 or self.config.get("wpm"):
            # Piper length_scale: 1.0 ≈ 200 wpm by default; scale to target.
            target_wpm = float(self.config.get("wpm", 200) or 200)
            length_scale = (200.0 / max(60.0, target_wpm)) / max(0.2, speed)
            cmd += ["--length_scale", f"{length_scale:.3f}"]
        try:
            proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return TTSResult(False, None, 0.0, self.name, voice_id, error=str(exc))
        if proc.returncode != 0 or not wav_tmp.exists():
            err = (proc.stderr or b"").decode("utf-8", "replace")[:400]
            return TTSResult(False, None, 0.0, self.name, voice_id, error=f"piper failed: {err}")
        wav_tmp.replace(out_path)
        return TTSResult(True, out_path, wav_duration(out_path), self.name, model)


class EspeakEngine(TTSEngine):
    """espeak-ng — offline, robotic, but very wide language coverage."""

    name = "espeak"

    _LANG_MAP = {
        "en": "en-us", "ur": "ur", "hi": "hi", "ar": "ar", "es": "es", "fr": "fr",
        "de": "de", "pt": "pt-br", "zh": "cmn", "ja": "ja", "ko": "ko", "tr": "tr",
        "ru": "ru", "bn": "bn", "pa": "pa", "it": "it", "id": "id", "fa": "fa", "he": "he",
    }

    def _bin(self) -> str | None:
        for cand in ("espeak-ng", "espeak-ng.exe", "espeak", "espeak.exe"):
            found = shutil.which(cand)
            if found:
                return found
        # Windows default install location
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        p = Path(pf) / "eSpeak NG" / "command_line_espeak.exe"
        if p.is_file():
            return str(p)
        p2 = Path(pf) / "eSpeak" / "command_line_espeak.exe"
        if p2.is_file():
            return str(p2)
        return None

    def probe(self) -> dict[str, Any]:
        binary = self._bin()
        ok = binary is not None
        return {
            "available": ok,
            "detail": f"espeak-ng at {binary}" if ok else "espeak-ng not found on PATH (install from https://github.com/espeak-ng/espeak-ng)",
            "binary": binary,
            "cost": "free",
            "account_required": False,
            "licensing": "GPL-3.0",
        }

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        if not self.probe()["available"]:
            return []
        langs = list(self._LANG_MAP.items())
        if language:
            key = _normalize_lang(language)
            langs = [(k, v) for k, v in langs if k == key]
        return [
            VoiceInfo(id=v, name=f"espeak {v}", language=v, engine=self.name, offline=True,
                      license_note="GPL-3.0")
            for _k, v in langs
        ]

    def synthesize(self, text, out_path, *, voice_id=None, speed=1.0, rate=DEFAULT_SAMPLE_RATE) -> TTSResult:
        info = self.probe()
        if not info["available"]:
            return TTSResult(False, None, 0.0, self.name, voice_id, error=info["detail"])
        lang = voice_id or "en-us"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # espeak -s is words-per-minute; use the project's target pace.
        target_wpm = float(self.config.get("wpm", 130) or 130)
        wpm = max(70, min(320, int(target_wpm * speed)))
        cmd = [info["binary"], "-v", lang, "-s", str(wpm), "-w", str(out_path), text]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return TTSResult(False, None, 0.0, self.name, voice_id, error=str(exc))
        if proc.returncode != 0 or not out_path.exists():
            err = (proc.stderr or b"").decode("utf-8", "replace")[:400]
            return TTSResult(False, None, 0.0, self.name, voice_id, error=f"espeak failed: {err}")
        return TTSResult(True, out_path, wav_duration(out_path), self.name, lang)


class SapiEngine(TTSEngine):
    """Windows Speech API via pyttsx3 — uses installed system voices."""

    name = "sapi"

    def probe(self) -> dict[str, Any]:
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            return {"available": False, "detail": "pyttsx3 not installed (pip install pyttsx3)",
                    "cost": "free", "account_required": False, "licensing": "n/a"}
        if os.name != "nt":
            return {"available": False, "detail": "SAPI is Windows-only",
                    "cost": "free", "account_required": False}
        return {
            "available": True,
            "detail": "Windows SAPI via pyttsx3",
            "cost": "free",
            "account_required": False,
            "licensing": "Uses voices installed with Windows; extra voices via Settings > Time & Language > Speech.",
        }

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        if os.name != "nt" or not self.probe()["available"]:
            return []
        try:
            import pyttsx3
            engine = pyttsx3.init()
            raw = engine.getProperty("voices") or []
            engine.stop()
        except Exception:  # noqa: BLE001
            return []
        out: list[VoiceInfo] = []
        for v in raw:
            vid = str(getattr(v, "id", ""))
            name = str(getattr(v, "name", vid))
            low = (name + " " + vid).lower()
            lang = "en-US"
            m = re.search(r"language[=:\s]+([a-z]{2}[-_][a-z]{2})", low)
            if m:
                lang = m.group(1).replace("_", "-")
            else:
                m2 = re.search(r"\b([a-z]{2})[-_]([a-z]{2})\b", vid)
                if m2:
                    lang = f"{m2.group(1)}-{m2.group(2)}"
            if language and _normalize_lang(lang) != _normalize_lang(language):
                continue
            out.append(VoiceInfo(id=vid, name=name, language=lang, engine=self.name,
                                 offline=True, license_note="System voice; license per Microsoft terms."))
        return out

    def synthesize(self, text, out_path, *, voice_id=None, speed=1.0, rate=DEFAULT_SAMPLE_RATE) -> TTSResult:
        if os.name != "nt":
            return TTSResult(False, None, 0.0, self.name, voice_id, error="SAPI is Windows-only")
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            return TTSResult(False, None, 0.0, self.name, voice_id, error=f"pyttsx3 init failed: {exc}")
        try:
            if voice_id:
                engine.setProperty("voice", voice_id)
            # SAPI rate is approximately words-per-minute (default ~200).
            target_wpm = float(self.config.get("wpm", 130) or 130)
            engine.setProperty("rate", max(70, min(320, int(target_wpm * speed))))
            engine.setProperty("volume", float(self.config.get("volume", 1.0)))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
            return TTSResult(False, None, 0.0, self.name, voice_id, error=f"sapi synthesis failed: {exc}")
        finally:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
        if not out_path.exists() or out_path.stat().st_size < 44:
            return TTSResult(False, None, 0.0, self.name, voice_id, error="sapi produced no audio")
        return TTSResult(True, out_path, wav_duration(out_path), self.name, voice_id)


class SilenceEngine(TTSEngine):
    """Emits correctly-timed silence so the pipeline still completes.

    Reported honestly: narration is marked as ``silence`` in the final report.
    """

    name = "silence"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.wpm = int(self.config.get("wpm", 130))

    def probe(self) -> dict[str, Any]:
        return {
            "available": True,
            "detail": "silent placeholder narration (timed to script length; no speech)",
            "cost": "free",
            "account_required": False,
            "licensing": "n/a",
        }

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        return [VoiceInfo(id="silence", name="No narration (silence)", language=language or "und",
                          engine=self.name, offline=True)]

    def synthesize(self, text, out_path, *, voice_id=None, speed=1.0, rate=DEFAULT_SAMPLE_RATE) -> TTSResult:
        seconds = _estimate_silence_seconds(text, int(self.wpm * speed))
        write_silence_wav(out_path, seconds, rate)
        return TTSResult(True, out_path, seconds, self.name, "silence",
                         notes=["Narration is silence — no TTS engine/voice was available."])


# ---------------------------------------------------------------------------
# Registry / facade
# ---------------------------------------------------------------------------
_ENGINE_CLASSES: dict[str, type[TTSEngine]] = {
    "piper": PiperEngine,
    "espeak": EspeakEngine,
    "sapi": SapiEngine,
    "silence": SilenceEngine,
}


@dataclass
class ProviderReport:
    engine: str
    available: bool
    detail: str
    voices: list[VoiceInfo] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "available": self.available,
            "detail": self.detail,
            "voices": [v.to_dict() for v in self.voices],
            **self.extra,
        }


def probe_all_engines() -> list[ProviderReport]:
    reports: list[ProviderReport] = []
    for name, cls in _ENGINE_CLASSES.items():
        eng = cls({})
        try:
            info = eng.probe()
        except Exception as exc:  # noqa: BLE001
            reports.append(ProviderReport(name, False, f"probe crashed: {exc}"))
            continue
        voices: list[VoiceInfo] = []
        if info.get("available"):
            try:
                voices = eng.list_voices()
            except Exception:  # noqa: BLE001
                voices = []
        extra = {k: v for k, v in info.items() if k not in ("available", "detail")}
        reports.append(ProviderReport(name, bool(info.get("available")), str(info.get("detail", "")), voices, extra))
    return reports


def pick_engine(preferred: str = "auto", *, language: str | None = None,
                config: dict[str, Any] | None = None) -> TTSEngine:
    """Pick the best available engine, honestly falling back to silence."""
    cfg = config or {}
    if preferred != "auto":
        cls = _ENGINE_CLASSES.get(preferred)
        if cls is None:
            raise TTSError(f"unknown TTS engine: {preferred}")
        eng = cls(cfg)
        if preferred != "silence" and not eng.probe().get("available"):
            raise TTSError(f"requested TTS engine '{preferred}' is not available")
        return eng

    # Auto: prefer engines that actually have a voice for the language.
    for name in ("piper", "espeak", "sapi"):
        eng = _ENGINE_CLASSES[name](cfg)
        info = eng.probe()
        if not info.get("available"):
            continue
        if language:
            try:
                voices = eng.list_voices(language)
            except Exception:  # noqa: BLE001
                voices = []
            if not voices:
                # No installed voice speaks this language; a mismatched voice
                # would read gibberish.  Fall through to the next engine (the
                # silence engine is the honest fallback).
                continue
        return eng
    return SilenceEngine(cfg)


def engine_by_name(name: str, config: dict[str, Any] | None = None) -> TTSEngine:
    """Return an engine instance by name (public wrapper over the registry)."""
    cls = _ENGINE_CLASSES.get(name)
    if cls is None:
        raise TTSError(f"unknown TTS engine: {name}")
    return cls(config or {})


def available_voices(language: str | None = None, *,
                     config: dict[str, Any] | None = None,
                     engines: Iterable[str] = ("piper", "espeak", "sapi")) -> list[VoiceInfo]:
    """Collect voices that are *actually installed* across engines.

    Never reports a voice merely because an engine is listed; each engine is
    probed first.  ``silence`` is never returned as a voice.
    """
    cfg = config or {}
    out: list[VoiceInfo] = []
    for name in engines:
        cls = _ENGINE_CLASSES.get(name)
        if cls is None:
            continue
        eng = cls(cfg)
        try:
            if not eng.probe().get("available"):
                continue
            out.extend(eng.list_voices(language))
        except Exception:  # noqa: BLE001 — a broken engine must not crash the caller
            continue
    return out


def synthesize_with_fallback(
    text: str,
    out_path: Path,
    *,
    preferred: str = "auto",
    language: str | None = None,
    voice_id: str | None = None,
    speed: float = 1.0,
    config: dict[str, Any] | None = None,
    attempts: int = 3,
) -> TTSResult:
    """Try the preferred engine, then fall through the chain to silence."""
    cfg = config or {}
    chain: list[TTSEngine] = []
    pre_error: str | None = None
    if preferred == "auto":
        for name in ("piper", "espeak", "sapi"):
            eng = _ENGINE_CLASSES[name](cfg)
            if eng.probe().get("available"):
                chain.append(eng)
        chain.append(SilenceEngine(cfg))
    else:
        try:
            chain.append(pick_engine(preferred, language=language, config=cfg))
        except TTSError as exc:
            chain.append(SilenceEngine(cfg))
            pre_error = str(exc)

    errors: list[str] = []
    if pre_error:
        errors.append(pre_error)

    for eng in chain:
        if eng.name == "silence":
            continue
        for _ in range(max(1, attempts)):
            try:
                res = eng.synthesize(text, out_path, voice_id=voice_id, speed=speed)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{eng.name}: {exc}")
                continue
            if res.ok:
                res.notes.extend(errors)
                return res
            errors.append(f"{eng.name}: {res.error}")
    # Final fallback: silence with honest note.
    res = SilenceEngine(cfg).synthesize(text, out_path, speed=speed)
    res.notes.extend(errors)
    res.notes.append("Fell back to silent narration; see notes for engine errors.")
    return res
