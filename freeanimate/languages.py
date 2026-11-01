"""Language registry and per-role language settings.

The language list is data-driven: ``config/languages.json`` can add or remove
languages without touching core application code.  Every language entry records:

* ``code``        — ISO-639-1 style code used internally
* ``name``        — English display name
* ``native``      — name written in the language itself
* ``rtl``         — right-to-left writing system?
* ``tts``         — best-effort TTS engine hints (verified at runtime, never trusted)
* ``font_hints``  — Windows font filenames likely to cover the script
* ``script``      — writing-system family used for font matching
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Built-in language table (also written to config/languages.json on first run
# so users can extend it).
# ---------------------------------------------------------------------------
BUILTIN_LANGUAGES: list[dict] = [
    {"code": "en", "name": "English", "native": "English", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "ur", "name": "Urdu", "native": "اردو", "rtl": True, "script": "arabic",
     "font_hints": ["arial.ttf", "Jameel Noori Nastaleeq.ttf", "NotoNastaliqUrdu-Regular.ttf", "Nirmala.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "hi", "name": "Hindi", "native": "हिन्दी", "rtl": False, "script": "devanagari",
     "font_hints": ["Nirmala.ttf", "mangal.ttf", "NotoSansDevanagari-Regular.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "ar", "name": "Arabic", "native": "العربية", "rtl": True, "script": "arabic",
     "font_hints": ["arial.ttf", "simsun.ttc", "NotoNaskhArabic-Regular.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "es", "name": "Spanish", "native": "Español", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "fr", "name": "French", "native": "Français", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "de", "name": "German", "native": "Deutsch", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "pt", "name": "Portuguese", "native": "Português", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "zh", "name": "Chinese", "native": "中文", "rtl": False, "script": "cjk",
     "font_hints": ["msyh.ttc", "simsun.ttc", "NotoSansCJK-Regular.ttc"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "ja", "name": "Japanese", "native": "日本語", "rtl": False, "script": "cjk",
     "font_hints": ["msgothic.ttc", "meiryob.ttc", "NotoSansJP-Regular.ttc"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "ko", "name": "Korean", "native": "한국어", "rtl": False, "script": "cjk",
     "font_hints": ["malgun.ttf", "NotoSansKR-Regular.ttc"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "tr", "name": "Turkish", "native": "Türkçe", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "ru", "name": "Russian", "native": "Русский", "rtl": False, "script": "cyrillic",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "bn", "name": "Bengali", "native": "বাংলা", "rtl": False, "script": "bengali",
     "font_hints": ["Nirmala.ttf", "Vrinda.ttf", "NotoSansBengali-Regular.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ / پنجابی", "rtl": False, "script": "gurmukhi",
     "font_hints": ["Nirmala.ttf", "NotoSansGurmukhi-Regular.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "it", "name": "Italian", "native": "Italiano", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "id", "name": "Indonesian", "native": "Bahasa Indonesia", "rtl": False, "script": "latin",
     "font_hints": ["arial.ttf", "segoeui.ttf"], "tts": ["sapi", "espeak", "piper"]},
    {"code": "fa", "name": "Persian", "native": "فارسی", "rtl": True, "script": "arabic",
     "font_hints": ["arial.ttf", "NotoNaskhArabic-Regular.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
    {"code": "he", "name": "Hebrew", "native": "עברית", "rtl": True, "script": "hebrew",
     "font_hints": ["arial.ttf", "Nirmala.ttf"],
     "tts": ["sapi", "espeak", "piper"]},
]

ROLE_NAMES = ("project", "script", "narration", "subtitle", "onscreen", "thumbnail")


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str
    rtl: bool
    script: str
    font_hints: tuple[str, ...] = field(default_factory=tuple)
    tts: tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        arrow = " (RTL)" if self.rtl else ""
        return f"{self.name} — {self.native}{arrow}"


def _to_language(entry: dict) -> Language:
    return Language(
        code=entry["code"],
        name=entry["name"],
        native=entry.get("native", entry["name"]),
        rtl=bool(entry.get("rtl", False)),
        script=entry.get("script", "latin"),
        font_hints=tuple(entry.get("font_hints", ())),
        tts=tuple(entry.get("tts", ())),
    )


class LanguageRegistry:
    """Ordered registry of supported languages backed by a JSON file."""

    def __init__(self, languages: Iterable[Language]):
        langs = list(languages)
        if not langs:
            raise ValueError("language registry must contain at least one language")
        self._by_code = {l.code: l for l in langs}
        self._ordered = langs

    # -- construction -------------------------------------------------------
    @classmethod
    def builtin(cls) -> "LanguageRegistry":
        return cls(_to_language(e) for e in BUILTIN_LANGUAGES)

    @classmethod
    def from_file(cls, path: Path | str | None = None) -> "LanguageRegistry":
        """Load from JSON, falling back to the built-in table.

        Malformed entries are skipped (reported via the ``errors`` list on the
        returned tuple in :meth:`load_with_errors`).
        """
        reg, _errors = cls.load_with_errors(path)
        return reg

    @classmethod
    def load_with_errors(cls, path: Path | str | None = None):
        if path is None:
            return cls.builtin(), []
        p = Path(path)
        if not p.exists():
            return cls.builtin(), [f"language file not found: {p}"]
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return cls.builtin(), [f"language file unreadable: {exc}"]
        entries = raw.get("languages", raw) if isinstance(raw, dict) else raw
        good: list[Language] = []
        errors: list[str] = []
        for i, e in enumerate(entries if isinstance(entries, list) else []):
            if not isinstance(e, dict) or "code" not in e or "name" not in e:
                errors.append(f"skipped malformed language entry #{i}")
                continue
            good.append(_to_language(e))
        if not good:
            errors.append("language file contained no valid entries; using built-ins")
            return cls.builtin(), errors
        return cls(good), errors

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"languages": [asdict(l) for l in self._ordered]}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- lookup -------------------------------------------------------------
    def __iter__(self):
        return iter(self._ordered)

    def __len__(self) -> int:
        return len(self._ordered)

    def __contains__(self, code: str) -> bool:
        return code in self._by_code

    @property
    def codes(self) -> list[str]:
        return [l.code for l in self._ordered]

    def get(self, code: str) -> Language | None:
        return self._by_code.get(code)

    def require(self, code: str) -> Language:
        lang = self.get(code)
        if lang is None:
            raise KeyError(f"unsupported language code: {code!r}")
        return lang

    def resolve(self, value: str | Language) -> Language:
        """Resolve a user-supplied selection (code, English name, or native
        name, case-insensitive) into a Language."""
        if isinstance(value, Language):
            return value
        v = value.strip()
        if not v:
            raise ValueError("empty language selection")
        low = v.lower()
        for lang in self._ordered:
            if lang.code == low or lang.name.lower() == low or lang.native == v or lang.native.lower() == low:
                return lang
        raise KeyError(f"unsupported language: {value!r}")

    def menu_lines(self) -> list[str]:
        """Lines for the interactive menu, numbered from 1."""
        return [f"[{i}] {lang.label()}" for i, lang in enumerate(self._ordered, start=1)]


@dataclass
class LanguageSettings:
    """Independent language settings per role (section 4.2).

    All roles default to the project language; individual roles may be
    overridden.
    """

    project: str = "en"
    script: str | None = None
    narration: str | None = None
    subtitle: str | None = None
    onscreen: str | None = None
    thumbnail: str | None = None

    def resolved(self, role: str) -> str:
        if role not in ROLE_NAMES:
            raise KeyError(f"unknown language role: {role}")
        value = getattr(self, role)
        return value or self.project

    def as_dict(self) -> dict:
        return {
            "project": self.project,
            "script": self.resolved("script"),
            "narration": self.resolved("narration"),
            "subtitle": self.resolved("subtitle"),
            "onscreen": self.resolved("onscreen"),
            "thumbnail": self.resolved("thumbnail"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LanguageSettings":
        proj = data.get("project", "en")
        return cls(
            project=proj,
            script=data.get("script"),
            narration=data.get("narration"),
            subtitle=data.get("subtitle"),
            onscreen=data.get("onscreen"),
            thumbnail=data.get("thumbnail"),
        )

    def uses_translation(self) -> bool:
        roles = [self.resolved(r) for r in ("script", "narration", "subtitle", "onscreen", "thumbnail")]
        return len(set(roles)) > 1 or roles[0] != self.project
