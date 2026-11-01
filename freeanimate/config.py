"""Configuration loading, validation, and defaults.

Configuration is a plain JSON document.  Invalid values raise
:class:`ConfigError` with an actionable message — never a silent fallback to a
different value (section 13: "Do not silently change requested settings").
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_AUDIO_CODEC,
    DEFAULT_CONTAINER,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_FPS,
    DEFAULT_MUSIC_VOLUME,
    DEFAULT_PIXEL_FORMAT,
    DEFAULT_RESOLUTION,
    DEFAULT_WPM,
    DEFAULT_X264_PRESET,
    DEFAULT_VIDEO_CODEC,
    FRAME_RATES,
    MAX_WPM,
    MIN_WPM,
    RESOLUTIONS,
)


class ConfigError(ValueError):
    """Raised when a configuration document is invalid."""


DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {
        "duration_minutes": 10,
        "resolution": DEFAULT_RESOLUTION,
        "fps": DEFAULT_FPS,
        "animation_mode": "hybrid",
        "transition": "fade",
        "project_language": "en",
        "script_language": None,
        "narration_language": None,
        "subtitle_language": None,
        "onscreen_language": None,
        "wpm": DEFAULT_WPM,
        "burn_subtitles": True,
        "music": "none",          # none | procedural | <path to audio file>
        "music_volume": DEFAULT_MUSIC_VOLUME,
        "low_resource": False,
        "duration_tolerance": DEFAULT_DURATION_TOLERANCE,
    },
    "render": {
        "video_codec": DEFAULT_VIDEO_CODEC,
        "audio_codec": DEFAULT_AUDIO_CODEC,
        "x264_preset": DEFAULT_X264_PRESET,
        "pixel_format": DEFAULT_PIXEL_FORMAT,
        "sample_rate": 44100,
        "container": DEFAULT_CONTAINER,
        "crf": 21,
    },
    "tts": {
        "engine": "auto",        # auto | sapi | espeak | piper | silence
        "voice": None,
        "speed": 1.0,            # multiplier
        "volume": 1.0,
        "chunk_chars": 1800,
    },
    "subtitles": {
        "font_size": 42,
        "margin_v": 112,
        "outline": 2,
        "shadow": 1,
        "max_line_chars": 46,
        "max_lines": 2,
    },
    "thumbnails": {
        "enabled": True,           # generate 3 thumbnails after a successful export
        "width": 1280,
        "height": 720,
        "quality": 90,             # JPEG quality
        "auto_select": False,      # copy recommended thumbnail to youtube_thumbnail.jpg
        "text_max_words": 5,       # concise thumbnail text budget
        "language": None,          # override thumbnail text language (defaults to project)
    },
    "paths": {
        "projects_dir": "projects",
        "cache_dir": "cache",
    },
    "providers": {
        "text": "template",      # template | ollama
        "translation": "none",   # none | mymemory (requires consent)
        "image": "procedural",   # procedural | (future backends)
        "video": "illustration", # illustration | (future AI backends)
    },
    "network": {
        "allow_translation_upload": False,  # must be explicitly enabled
    },
    "workers": 1,
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


_VALID_TRANSITIONS = {"fade", "crossfade", "none"}
_VALID_ANIMATION_MODES = {"illustration", "motion_graphics", "hybrid", "ai_video"}
_VALID_MUSIC = {"none", "procedural"}


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_CONFIG))
    source_path: Path | None = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path | str | None = None, *, create_if_missing: bool = True) -> "Config":
        p = Path(path) if path else None
        if p is not None and p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"config file {p} is not valid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ConfigError(f"config file {p} must contain a JSON object")
            merged = _deep_merge(DEFAULT_CONFIG, raw)
            cfg = cls(data=merged, source_path=p)
        else:
            cfg = cls(data=deepcopy(DEFAULT_CONFIG), source_path=p)
            if p is not None and create_if_missing:
                cfg.save(p)
        cfg.validate()
        return cfg

    def save(self, path: Path | str | None = None) -> Path:
        p = Path(path) if path else self.source_path
        if p is None:
            raise ConfigError("no path given to save configuration")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.source_path = p
        return p

    # -------------------------------------------------------------- validate
    def validate(self) -> None:
        d = self.data
        for section in ("defaults", "render", "tts", "subtitles", "paths", "providers", "network"):
            if section not in d or not isinstance(d[section], dict):
                raise ConfigError(f"missing or invalid config section: {section!r}")

        defaults = d["defaults"]
        dur = defaults.get("duration_minutes")
        if not isinstance(dur, (int, float)) or dur <= 0:
            raise ConfigError("defaults.duration_minutes must be a positive number")

        res = defaults.get("resolution")
        if res not in RESOLUTIONS:
            raise ConfigError(f"defaults.resolution must be one of {sorted(RESOLUTIONS)}, got {res!r}")

        fps = defaults.get("fps")
        if fps not in FRAME_RATES:
            raise ConfigError(f"defaults.fps must be one of {FRAME_RATES}, got {fps!r}")

        mode = defaults.get("animation_mode")
        if mode not in _VALID_ANIMATION_MODES:
            raise ConfigError(f"defaults.animation_mode must be one of {sorted(_VALID_ANIMATION_MODES)}")

        trans = defaults.get("transition")
        if trans not in _VALID_TRANSITIONS:
            raise ConfigError(f"defaults.transition must be one of {sorted(_VALID_TRANSITIONS)}")

        wpm = defaults.get("wpm")
        if not isinstance(wpm, (int, float)) or not (MIN_WPM <= wpm <= MAX_WPM):
            raise ConfigError(f"defaults.wpm must be between {MIN_WPM} and {MAX_WPM}")

        tol = defaults.get("duration_tolerance")
        if not isinstance(tol, (int, float)) or not (0.0 <= tol < 1.0):
            raise ConfigError("defaults.duration_tolerance must be between 0.0 and 1.0")

        music = defaults.get("music", "none")
        if not (music in _VALID_MUSIC or (isinstance(music, str) and music)):
            raise ConfigError("defaults.music must be 'none', 'procedural', or a file path")

        mv = defaults.get("music_volume")
        if not isinstance(mv, (int, float)) or not (0.0 <= mv <= 1.0):
            raise ConfigError("defaults.music_volume must be between 0.0 and 1.0")

        render = d["render"]
        crf = render.get("crf")
        if not isinstance(crf, int) or not (0 <= crf <= 51):
            raise ConfigError("render.crf must be an integer between 0 and 51")
        sr = render.get("sample_rate")
        if sr not in (22050, 32000, 44100, 48000):
            raise ConfigError("render.sample_rate must be one of 22050, 32000, 44100, 48000")
        container = render.get("container")
        if container not in ("mp4", "mkv", "mov"):
            raise ConfigError("render.container must be mp4, mkv, or mov")

        tts = d["tts"]
        if tts.get("engine") not in ("auto", "sapi", "espeak", "piper", "silence"):
            raise ConfigError("tts.engine must be auto, sapi, espeak, piper, or silence")
        sp = tts.get("speed")
        if not isinstance(sp, (int, float)) or not (0.5 <= sp <= 2.0):
            raise ConfigError("tts.speed must be between 0.5 and 2.0")

        if not isinstance(d["network"].get("allow_translation_upload", False), bool):
            raise ConfigError("network.allow_translation_upload must be a boolean")

        thumbs = d.get("thumbnails", {})
        if not isinstance(thumbs, dict):
            raise ConfigError("thumbnails section must be a JSON object")
        if not isinstance(thumbs.get("enabled", True), bool):
            raise ConfigError("thumbnails.enabled must be a boolean")
        tq = thumbs.get("quality", 90)
        if not isinstance(tq, int) or not (50 <= tq <= 100):
            raise ConfigError("thumbnails.quality must be an integer between 50 and 100")
        if not isinstance(thumbs.get("auto_select", False), bool):
            raise ConfigError("thumbnails.auto_select must be a boolean")
        for dim in ("width", "height"):
            dv = thumbs.get(dim)
            if dv is not None and (not isinstance(dv, int) or dv < 320 or dv > 7680):
                raise ConfigError(f"thumbnails.{dim} must be an integer between 320 and 7680")

        workers = d.get("workers", 1)
        if not isinstance(workers, int) or workers < 1 or workers > 8:
            raise ConfigError("workers must be an integer between 1 and 8")

    # ------------------------------------------------------------------ get
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {dotted}: {part} is not a section")
        node[parts[-1]] = value
        self.validate()
